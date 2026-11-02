"""契约往返测试：样例 → 模型 → 再序列化，必须与样例逐字段一致。

为什么需要这组测试：
    现有 test_question_officer.py 只做 model_validate(样例)，是**单向**的——
    只守「读得进来」，守不住「吐得出去」。而 Java 侧消费的是 **Python 序列化
    出来的 JSON**，不是样例文件本身。

    会漏掉的典型事故：给模型加一个带默认值的新字段，validate 照样通过（旧的
    单向测试全绿），但 model_dump() 会多吐一个字段 → Java 严格模式
    （FAIL_ON_UNKNOWN_PROPERTIES=true）直接解析失败。

覆盖边界：
    能抓 —— 字段名改动、类型改动、多字段/少字段、枚举取值被改。
    抓不到 —— 字段「值」是否语义正确（例如样例里模型名写成旧名）、业务规则
    是否正确。那类问题靠 R11 联动与人工审查，不靠测试。
"""
import json
from pathlib import Path

import pytest

from app.schemas.health import HealthResponse
from app.schemas.question_officer import QuestionOfficerRequest, QuestionOfficerResponse

CONTRACT_DIR = Path(__file__).resolve().parents[4] / "docs" / "server" / "contracts"

# (样例文件, 对应模型)；新增契约样例只需在此加一行
CASES = [
    ("health_resp.json", HealthResponse),
    ("question_officer_req.json", QuestionOfficerRequest),
    ("question_officer_resp.json", QuestionOfficerResponse),
    ("question_officer_smalltalk.json", QuestionOfficerResponse),
]


def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename,model_cls", CASES, ids=[c[0] for c in CASES])
def test_sample_roundtrip(filename: str, model_cls):
    """样例里出现过的每个键，序列化后必须还在且值相等。

    只遍历样例的键（不要求全等）：样例是「最小充分集」，模型可能有默认值字段
    不在样例里，全等比对会假失败。
    """
    sample = _load(filename)
    model = model_cls.model_validate(sample)

    # mode="json" 是必须的：默认 model_dump 会把 Enum 输出成枚举对象、
    # datetime 输出成 datetime 对象，而样例里都是字符串，直接比会假失败
    dumped = model.model_dump(mode="json")

    for key, expected in sample.items():
        assert key in dumped, f"序列化后丢失字段：{key}"
        assert dumped[key] == expected, (
            f"字段 {key} 序列化不一致\n  实际: {dumped[key]!r}\n  样例: {expected!r}"
        )


@pytest.mark.parametrize("filename,model_cls", CASES, ids=[c[0] for c in CASES])
def test_no_extra_fields_in_output(filename: str, model_cls):
    """模型不得吐出样例之外的字段（Java 严格模式会因未知字段解析失败）。

    这条守的是「加字段炸下游」：Python 新增带默认值的字段时，旧单向测试不会红，
    这里会。若新增字段是契约的一部分，须同步补齐样例与 Java DTO。
    """
    sample = _load(filename)
    dumped = model_cls.model_validate(sample).model_dump(mode="json")

    extra = set(dumped) - set(sample)
    assert not extra, (
        f"序列化多出样例没有的字段（Java 严格模式会解析失败）：{sorted(extra)}\n"
        f"  契约样例: {filename}\n"
        f"  处理方式: 若该字段属于契约 → 补进样例并同步 Java DTO；否则从模型移除"
    )
