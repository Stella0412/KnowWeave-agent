from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.prompt_service import CHAT_PROMPT


def test_chat_prompt_assembles_messages():
    history = [HumanMessage(content="q1"), AIMessage(content="a1")]
    messages = CHAT_PROMPT.invoke({"history": history, "input": "q2"}).to_messages()
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].content == "q1"
    assert messages[2].content == "a1"
    assert messages[3].content == "q2"  # 当前输入排最后
    assert isinstance(messages[3], HumanMessage)


def test_system_prompt_names_the_mall():
    assert "知序" in SYSTEM_PROMPT
    assert "不编造" in SYSTEM_PROMPT


def test_system_prompt_has_citation_and_refusal_sections():
    """ch04 新增三节落在【行为约束】之前，且既有约束那句仍在（:20 那条依赖它）。

    行为验证在 T22（忠实度裁判规则 3/5 + 负面断言检查）；这里只钉结构：
    节标题在场、顺序对、且插入没有吃掉既有内容。
    """
    for section in ("【引用规范】", "【拒答规范】", "【禁止承诺的负面知识】"):
        assert section in SYSTEM_PROMPT
    # 顺序：三节都在【行为约束】之前（不能插到文件末尾、也不能插进别的节里）
    assert SYSTEM_PROMPT.index("【引用规范】") < SYSTEM_PROMPT.index("【行为约束】")
    assert SYSTEM_PROMPT.index("【拒答规范】") < SYSTEM_PROMPT.index("【行为约束】")
    assert SYSTEM_PROMPT.index("【禁止承诺的负面知识】") < SYSTEM_PROMPT.index("【行为约束】")
    # 插入没吃掉既有内容
    assert "不编造订单状态、物流信息、退款进度" in SYSTEM_PROMPT
