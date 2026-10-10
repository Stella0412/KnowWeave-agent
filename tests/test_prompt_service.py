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
