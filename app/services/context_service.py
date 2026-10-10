import tiktoken
from langchain_core.messages import BaseMessage


class ContextStore:
    """纯 token 预算裁剪器：不再持有会话状态（DB 是唯一真相源），只做计数与裁剪。"""

    def __init__(self, token_budget: int = 4096):
        self._token_budget = token_budget
        self._enc = tiktoken.get_encoding("cl100k_base")  # 近似计数，DeepSeek 够用

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for m in messages:
            total += 4  # 每条消息协议开销（近似）
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += len(self._enc.encode(content))
        return total

    def trim(self, system_message: BaseMessage, history: list[BaseMessage]) -> list[BaseMessage]:
        """返回裁剪后的历史（不含 system），使 system+history 不超预算，且保留最新一条。"""
        history = list(history)  # 不修改调用方
        while (
            len(history) > 1
            and self.count_tokens([system_message] + history) > self._token_budget
        ):
            history.pop(0)
        return history
