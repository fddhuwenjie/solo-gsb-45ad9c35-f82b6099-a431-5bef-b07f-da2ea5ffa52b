"""设备批次链一致性核验测试（只读快照，不改数据）。

覆盖：健康链（未归档/部分归档/全归档/空范围外全局项）逐批复算通过、
previous_digest 链与序号连续性、链游标 batch_count/last_sequence/head_digest、
archived_through 墓碑边界、消费者确认摘要与 watermark 重算、
batch_judgment_batches 来源摘要；各类篡改只进问题清单（ok=false）且不改数据；
空设备 404、区间含缺批次 404、非法范围 422；并发上传/确认/归档期间结果来自
同一快照（健康状态恒为 ok）。
"""
import json
import os
import sqlite3
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

THRESHOLDS = {"tolerance": 0.5, "max_drift": 5e-4, "max_missed": 3, "max_false": 3}
DEVICE = "dev-01"
VERIFY = f"/field-devices/{DEVICE}/chain-verification"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    app.state.db_file = str(tmp_path / "test.db")
    with TestClient(app) as c:
        yield c


def obs(n, base=1_700_000_000.0, step=0.5):
    return [
        {"local_id": f"o{i:03d}", "captured_at": base + i * step}
        for i in range(n)
    ]


def batch(batch_id, sequence, previous_digest="", observations=None, **kw):
    body = {
        "batch_id": batch_id,
        "sequence": sequence,
        "previous_digest": previous_digest,
        "observations": observations if observations is not None else obs(3),
        **THRESHOLDS,
    }
    body.update(kw)
    return body


def upload_chain(client, device, n):
    """上传 n 个链式批次，返回各批摘要（下标 0 = seq1）。"""
    digests = []
    prev = ""
    for seq in range(1, n + 1):
        r = client.post(
            f"/field-devices/{device}/batches",
            json=batch(f"b-{seq}", seq, previous_digest=prev),
        )
        assert r.status_code == 201
        prev = r.json()["digest"]
        digests.append(prev)
    return digests


def verify(client, frm, to, device=DEVICE):
    return client.post(
        f"/field-devices/{device}/chain-verification",
        json={"from_sequence": frm, "to_sequence": to},
    )


def issues_by_code(payload):
    return {i["issue_code"]: i for i in payload["issues"]}


# ---------------- 健康链：核验通过 ----------------


def test_healthy_chain_verifies_ok(client):
    digests = upload_chain(client, DEVICE, 4)
    r = verify(client, 1, 4)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["issue_count"] == 0 and body["issues"] == []
    assert body["from_sequence"] == 1 and body["to_sequence"] == 4
    # 快照上限 = 核验开始时链头序号
    assert body["snapshot_head_sequence"] == 4
    snap = body["snapshot"]
    assert snap["head_sequence"] == 4
    assert snap["head_digest"] == digests[-1]
    assert snap["batch_count"] == 4
    assert snap["archived_through"] == 0
    assert snap["consumer_count"] == 0
    assert snap["snapshot_at"]


def test_verify_subrange_only_recomputes_that_range(client):
    upload_chain(client, DEVICE, 5)
    r = verify(client, 2, 3)
    assert r.status_code == 200
    body = r.json()
    # 链游标为全局项，无论区间如何都核对；区间逐批项只覆盖 2..3
    assert body["ok"] is True
    assert body["from_sequence"] == 2 and body["to_sequence"] == 3
    assert body["snapshot_head_sequence"] == 5


def test_healthy_single_first_batch_previous_empty(client):
    upload_chain(client, DEVICE, 1)
    body = verify(client, 1, 1).json()
    assert body["ok"] is True


def test_verify_with_consumers_and_watermarks_ok(client):
    digests = upload_chain(client, DEVICE, 3)
    for cid in ("archiver", "ml"):
        assert client.post(
            f"/field-devices/{DEVICE}/consumers",
            json={"consumer_id": cid},
        ).status_code in (200, 201)
    # archiver 乱序确认 1,2,3；ml 只确认 1,3（水位停在 1）
    for seq in (1, 2, 3):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "archiver", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "ml", "sequence": 1, "batch_digest": digests[0]},
    )
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "ml", "sequence": 3, "batch_digest": digests[2]},
    )
    body = verify(client, 1, 3).json()
    assert body["ok"] is True
    assert body["snapshot"]["consumer_count"] == 2


def test_verify_archived_prefix_ok(client):
    digests = upload_chain(client, DEVICE, 4)
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "arch"})
    for seq in range(1, 5):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "arch", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    r = client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 2})
    assert r.status_code == 201

    # 全区间：1..2 走墓碑核对，3..4 重算，全部通过
    body = verify(client, 1, 4).json()
    assert body["ok"] is True
    assert body["snapshot"]["archived_through"] == 2

    # 只核验墓碑区间同样通过（不要求明细）
    body = verify(client, 1, 2).json()
    assert body["ok"] is True

    # 归档后登记的消费者从 archived_through 起步，水位重算仍一致
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "late"})
    body = verify(client, 1, 4).json()
    assert body["ok"] is True
    assert body["snapshot"]["consumer_count"] == 2


def test_verify_with_batch_judgment_sources_ok(client):
    digests = upload_chain(client, DEVICE, 3)
    r = client.post(
        f"/field-devices/{DEVICE}/batch-judgments",
        json={"idempotency_key": "k1", "from_sequence": 1, "to_sequence": 3},
    )
    assert r.status_code == 201
    body = verify(client, 1, 3).json()
    assert body["ok"] is True


# ---------------- 4xx：空设备 / 缺批次 / 非法范围 ----------------


def test_verify_empty_device_returns_404(client):
    r = verify(client, 1, 1, device="dev-unknown")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "device_not_found"
    assert detail["device_id"] == "dev-unknown"


def test_verify_range_beyond_head_returns_404(client):
    upload_chain(client, DEVICE, 2)
    r = verify(client, 1, 3)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "verification_range_not_found"
    assert detail["missing_sequences"] == [3]
    assert detail["head_sequence"] == 2


def test_verify_range_with_internal_gap_returns_404(client):
    # 正常 API 造不出链内缺口；直接造一条缺口链
    db_file = client.app.state.db_file
    upload_chain(client, DEVICE, 3)
    with sqlite3.connect(db_file) as conn:
        conn.execute("DELETE FROM device_batches WHERE device_id=? AND sequence=2", (DEVICE,))
        conn.execute(
            "UPDATE device_chains SET batch_count = batch_count - 1 WHERE device_id=?",
            (DEVICE,),
        )
        conn.commit()
    r = verify(client, 1, 3)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["missing_sequences"] == [2]
    assert detail["head_sequence"] == 3


def test_verify_invalid_range_returns_422(client):
    upload_chain(client, DEVICE, 2)
    r = client.post(VERIFY, json={"from_sequence": 3, "to_sequence": 2})
    assert r.status_code == 422
    r = client.post(VERIFY, json={"from_sequence": 0, "to_sequence": 2})
    assert r.status_code == 422


# ---------------- 篡改检测：问题清单且不改数据 ----------------


def _tamper(db_file, sql, params=()):
    with sqlite3.connect(db_file) as conn:
        conn.execute(sql, params)
        conn.commit()


def test_detects_recalculated_digest_mismatch(client):
    upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    before = client.get(f"/field-devices/{DEVICE}/batches/2").json()
    # 篡改已保存明细（观测时间戳），但 digest 列不动 → 重算不符
    rows = _load_rows(db_file)
    observations = rows[2]["observations"]
    observations[0]["captured_at"] += 10.0
    _tamper(
        db_file,
        "UPDATE device_batches SET observations=? WHERE device_id=? AND sequence=2",
        (json.dumps(observations), DEVICE),
    )
    r = verify(client, 1, 3)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    issue = issues_by_code(body)["batch_digest_mismatch"]
    assert issue["sequence"] == 2
    assert issue["actual"]  # 保存的（旧）摘要
    assert issue["expected"] != issue["actual"]

    # 只读：核验未修复也未回滚篡改，明细仍是被篡改后的值，其余数据不变
    after = client.get(f"/field-devices/{DEVICE}/batches/2").json()
    assert after["observations"][0]["captured_at"] != before["observations"][0]["captured_at"]
    assert client.get(f"/field-devices/{DEVICE}/batches/1").json()["digest"] == rows[1]["digest"]


def _load_rows(db_file):
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        out = {}
        for r in conn.execute(
            "SELECT * FROM device_batches WHERE device_id=? ORDER BY sequence",
            (DEVICE,),
        ):
            d = dict(r)
            if d.get("observations"):
                d["observations"] = json.loads(d["observations"])
            if d.get("thresholds"):
                d["thresholds"] = json.loads(d["thresholds"])
            out[d["sequence"]] = d
        return out


def test_detects_previous_digest_chain_break(client):
    upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_batches SET previous_digest=? WHERE device_id=? AND sequence=2",
        ("00" * 32, DEVICE),
    )
    body = verify(client, 1, 3).json()
    assert body["ok"] is False
    issue = issues_by_code(body)["previous_digest_mismatch"]
    assert issue["sequence"] == 2
    assert issue["actual"] == "00" * 32
    # 期望为 seq1 保存摘要
    rows = _load_rows(db_file)
    assert issue["expected"] == rows[1]["digest"]


def test_detects_first_batch_previous_digest_nonempty(client):
    upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_batches SET previous_digest=? WHERE device_id=? AND sequence=1",
        ("ab" * 32, DEVICE),
    )
    body = verify(client, 1, 2).json()
    issue = issues_by_code(body)["previous_digest_mismatch"]
    assert issue["sequence"] == 1 and issue["expected"] == ""
    assert issue["actual"] == "ab" * 32


def test_detects_sequence_gap_outside_range(client):
    db_file = client.app.state.db_file
    upload_chain(client, DEVICE, 3)
    with sqlite3.connect(db_file) as conn:
        conn.execute("DELETE FROM device_batches WHERE device_id=? AND sequence=2", (DEVICE,))
        conn.commit()
    # 只核验 seq3（区间本身完整）：链级 sequence_gap 仍须上报
    body = verify(client, 3, 3).json()
    assert body["ok"] is False
    issue = issues_by_code(body)["sequence_gap"]
    assert issue["sequence"] == 2


def test_detects_cursor_fields(client):
    upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_chains SET last_sequence=9, batch_count=7, head_digest=? "
        "WHERE device_id=?",
        ("11" * 32, DEVICE),
    )
    body = verify(client, 1, 3).json()
    by_code = issues_by_code(body)
    assert by_code["last_sequence_mismatch"]["actual"] == 9
    assert by_code["last_sequence_mismatch"]["expected"] == 3
    assert by_code["batch_count_mismatch"]["actual"] == 7
    assert by_code["batch_count_mismatch"]["expected"] == 3
    assert by_code["head_digest_mismatch"]["actual"] == "11" * 32
    assert body["ok"] is False


def test_detects_archived_boundary_mismatch(client):
    digests = upload_chain(client, DEVICE, 4)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in range(1, 5):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 2})

    # 篡改链游标 archived_through 与实际墓碑不符（3 未归档）
    _tamper(
        db_file,
        "UPDATE device_chains SET archived_through=3 WHERE device_id=?",
        (DEVICE,),
    )
    body = verify(client, 1, 4).json()
    issue = issues_by_code(body)["archived_boundary_mismatch"]
    assert issue["expected"] == 2
    # actual 给出游标值与实际墓碑序号：3 声称已归档但墓碑只有 1,2
    assert issue["actual"] == {
        "archived_through": 3,
        "tombstone_sequences": [1, 2],
    }

    # 越界值（> last_sequence）也能识别
    _tamper(
        db_file,
        "UPDATE device_chains SET archived_through=99 WHERE device_id=?",
        (DEVICE,),
    )
    body = verify(client, 1, 4).json()
    issue = issues_by_code(body)["archived_boundary_mismatch"]
    assert issue["expected"].startswith("0 <= archived_through")
    assert issue["actual"] == 99


def test_detects_tombstone_state_mismatch(client):
    digests = upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in range(1, 4):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 2})
    # 墓碑 status 被改回 accepted，但明细仍为空 → 墓碑状态不一致
    _tamper(
        db_file,
        "UPDATE device_batches SET status='accepted' WHERE device_id=? AND sequence=1",
        (DEVICE,),
    )
    body = verify(client, 1, 3).json()
    issue = issues_by_code(body)["tombstone_state_mismatch"]
    assert issue["sequence"] == 1
    assert issue["expected"] == "archived"
    assert issue["actual"] == "accepted"


def test_archived_batch_digest_change_detected(client):
    """归档批次只核对保存摘要：确认记录引用旧摘要，摘要列被篡改即不一致。"""
    digests = upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in (1, 2):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 1})
    _tamper(
        db_file,
        "UPDATE device_batches SET digest=? WHERE device_id=? AND sequence=1",
        ("77" * 32, DEVICE),
    )
    body = verify(client, 1, 2).json()
    by_code = issues_by_code(body)
    # seq2.previous_digest 仍引用旧摘要 → 链断裂
    assert by_code["previous_digest_mismatch"]["sequence"] == 2
    # head/archived 边界本身不依赖 digest；确认摘要与被篡改的保存摘要不符
    ack = by_code["ack_digest_mismatch"]
    assert ack["sequence"] == 1 and ack["consumer_id"] == "a"
    assert ack["expected"] == "77" * 32 and ack["actual"] == digests[0]


def test_ack_digest_checked_globally_outside_range(client):
    """确认摘要为链级不变量：只核验其他区间时，区间外的错误确认仍须上报。"""
    digests = upload_chain(client, DEVICE, 4)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "a", "sequence": 1, "batch_digest": digests[0]},
    )
    _tamper(
        db_file,
        "UPDATE device_batch_acks SET batch_digest=? WHERE device_id=? AND sequence=1",
        ("99" * 32, DEVICE),
    )
    body = verify(client, 3, 4).json()
    issue = issues_by_code(body)["ack_digest_mismatch"]
    assert issue["sequence"] == 1 and issue["consumer_id"] == "a"


def test_detects_ack_digest_mismatch_on_live_batch(client):
    digests = upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "a", "sequence": 1, "batch_digest": digests[0]},
    )
    # 直接篡改确认摘要（正常 API 不允许写入错摘要）
    _tamper(
        db_file,
        "UPDATE device_batch_acks SET batch_digest=? WHERE device_id=? AND sequence=1",
        ("99" * 32, DEVICE),
    )
    body = verify(client, 1, 2).json()
    issue = issues_by_code(body)["ack_digest_mismatch"]
    assert issue["sequence"] == 1 and issue["consumer_id"] == "a"
    assert issue["expected"] == digests[0] and issue["actual"] == "99" * 32


def test_detects_watermark_mismatch(client):
    digests = upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in (1, 2, 3):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    # 缓存水位与稀疏确认不一致
    _tamper(
        db_file,
        "UPDATE device_consumers SET watermark=1 WHERE device_id=? AND consumer_id='a'",
        (DEVICE,),
    )
    body = verify(client, 1, 3).json()
    issue = issues_by_code(body)["watermark_mismatch"]
    assert issue["consumer_id"] == "a"
    assert issue["expected"] == 3 and issue["actual"] == 1


def test_watermark_with_sparse_acks_and_archived_base(client):
    digests = upload_chain(client, DEVICE, 4)
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in range(1, 5):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )
    client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 2})
    # 迟到消费者：水位从 archived_through=2 起步，再确认 seq4（seq3 缺口）
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "late"})
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "late", "sequence": 4, "batch_digest": digests[3]},
    )
    body = verify(client, 1, 4).json()
    assert body["ok"] is True  # late 水位 2，缺 seq3 不推进——重算规则一致


def test_detects_batch_judgment_source_mismatch(client):
    digests = upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file
    client.post(
        f"/field-devices/{DEVICE}/batch-judgments",
        json={"idempotency_key": "k1", "from_sequence": 1, "to_sequence": 3},
    )
    # 篡改已保存来源中的某批摘要
    _tamper(
        db_file,
        "UPDATE batch_judgment_batches SET digest=? WHERE sequence=2",
        ("ee" * 32,),
    )
    body = verify(client, 1, 3).json()
    issue = issues_by_code(body)["batch_judgment_source_mismatch"]
    assert issue["sequence"] == 2 and issue["field"] == "digest"
    assert issue["actual"] == "ee" * 32 and issue["expected"] == digests[1]


def test_detects_observation_count_mismatch(client):
    upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_batches SET observation_count=99 WHERE device_id=? AND sequence=1",
        (DEVICE,),
    )
    body = verify(client, 1, 2).json()
    issue = issues_by_code(body)["observation_count_mismatch"]
    assert issue["sequence"] == 1 and issue["expected"] == 3 and issue["actual"] == 99


def test_live_batch_with_null_details_reported_not_crashed(client):
    """非墓碑批次明细被清空（损坏）：报墓碑状态问题，核验本身不抛异常。"""
    upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_batches SET observations=NULL, thresholds=NULL "
        "WHERE device_id=? AND sequence=1",
        (DEVICE,),
    )
    r = verify(client, 1, 2)
    assert r.status_code == 200
    issue = issues_by_code(r.json())["tombstone_state_mismatch"]
    assert issue["sequence"] == 1
    assert issue["actual"]["observations"] is None


def test_issues_have_required_shape(client):
    upload_chain(client, DEVICE, 2)
    db_file = client.app.state.db_file
    _tamper(
        db_file,
        "UPDATE device_batches SET previous_digest=? WHERE device_id=? AND sequence=2",
        ("00" * 32, DEVICE),
    )
    body = verify(client, 1, 2).json()
    assert body["issue_count"] == 1
    issue = body["issues"][0]
    assert set(issue) >= {"issue_code", "sequence", "expected", "actual"}


# ---------------- 只读：核验绝不改数据 ----------------


def test_verification_writes_nothing(client):
    digests = upload_chain(client, DEVICE, 3)
    db_file = client.app.state.db_file

    def snapshot_db():
        with sqlite3.connect(db_file) as conn:
            tables = (
                "device_chains", "device_batches", "device_consumers",
                "device_batch_acks", "batch_judgments", "batch_judgment_batches",
            )
            out = {}
            for t in tables:
                out[t] = conn.execute(
                    f"SELECT * FROM {t} ORDER BY 1"
                ).fetchall()
            return out

    before = snapshot_db()
    for frm, to in ((1, 1), (1, 3), (2, 2)):
        verify(client, frm, to)
    assert snapshot_db() == before
    # 业务数据仍可正常读取
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 3 and cur["head_digest"] == digests[-1]


# ---------------- 并发：结果必须来自同一快照 ----------------


def test_concurrent_mutations_during_verification_always_consistent(client):
    """上传/确认与核验并发：API 维护的状态在任一快照下都必须核验 ok。"""
    digests = upload_chain(client, DEVICE, 1)
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "a", "sequence": 1, "batch_digest": digests[0]},
    )

    n = 20
    errors = []
    stop = threading.Event()
    lock = threading.Lock()

    def mutator():
        prev = digests[0]
        for seq in range(2, n + 1):
            r = client.post(
                f"/field-devices/{DEVICE}/batches",
                json=batch(f"b-{seq}", seq, previous_digest=prev),
            )
            assert r.status_code == 201
            prev = r.json()["digest"]
            ar = client.post(
                f"/field-devices/{DEVICE}/acknowledgments",
                json={"consumer_id": "a", "sequence": seq, "batch_digest": prev},
            )
            assert ar.status_code in (200, 201)

    def verifier():
        while not stop.is_set():
            cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
            head = cur["last_sequence"]
            if head < 1:
                continue
            r = verify(client, 1, head)
            if r.status_code != 200:
                with lock:
                    errors.append(("status", r.status_code, r.text))
                continue
            body = r.json()
            # 同快照：区间必须完整（并发上传的新批次不在本次区间内）
            if not body["ok"]:
                with lock:
                    errors.append(("issues", body["issues"]))

    t = threading.Thread(target=mutator)
    vs = [threading.Thread(target=verifier) for _ in range(3)]
    t.start()
    for v in vs:
        v.start()
    t.join()
    stop.set()
    for v in vs:
        v.join()
    assert errors == []

    # 最终状态核验通过
    body = verify(client, 1, n).json()
    assert body["ok"] is True
    assert body["snapshot"]["head_sequence"] == n


def test_concurrent_archive_during_verification_always_consistent(client):
    n = 12
    digests = upload_chain(client, DEVICE, n)
    client.post(f"/field-devices/{DEVICE}/consumers", json={"consumer_id": "a"})
    for seq in range(1, n + 1):
        client.post(
            f"/field-devices/{DEVICE}/acknowledgments",
            json={"consumer_id": "a", "sequence": seq,
                  "batch_digest": digests[seq - 1]},
        )

    errors = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def archiver():
        barrier.wait()
        for through in range(1, n + 1):
            r = client.post(
                f"/field-devices/{DEVICE}/archive",
                json={"through_sequence": through},
            )
            assert r.status_code in (200, 201)

    def verifier():
        barrier.wait()
        for _ in range(n * 3):
            r = verify(client, 1, n)
            body = r.json()
            if r.status_code != 200 or not body["ok"]:
                with lock:
                    errors.append((r.status_code, body.get("issues")))

    threads = [threading.Thread(target=archiver), threading.Thread(target=verifier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    body = verify(client, 1, n).json()
    assert body["ok"] is True
    assert body["snapshot"]["archived_through"] == n


def test_snapshot_fixed_against_concurrent_append(client):
    """核验快照上限在事务开始即固定：响应 head 与逐批结果来自同一版本。"""
    upload_chain(client, DEVICE, 3)
    # 单次核验前快照为 3；核验后再上传不改变已返回结果
    body = verify(client, 1, 3).json()
    assert body["snapshot_head_sequence"] == 3
    prev = body["snapshot"]["head_digest"]
    r = client.post(
        f"/field-devices/{DEVICE}/batches",
        json=batch("b-4", 4, previous_digest=prev),
    )
    assert r.status_code == 201
    assert body["snapshot_head_sequence"] == 3
    assert body["snapshot"]["batch_count"] == 3
