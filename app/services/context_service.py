import tiktoken
from langchain_core.messages import BaseMessage


class ContextStore:
    """内存会话上下文：按 session_id 存消息，支持按 token 预算从最旧开始裁剪。"""

    def __init__(self, token_budget: int = 4096):
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._token_budget = token_budget
        self._enc = tiktoken.get_encoding("cl100k_base")  # 近似计数，DeepSeek 够用

    def append(self, session_id: str, message: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for m in messages:
            total += 4  # 每条消息协议开销（近似）
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += len(self._enc.encode(content))
        return total

    def trim_history(
        self, session_id: str, system_message: BaseMessage
    ) -> list[BaseMessage]:
        """返回裁剪后的历史（不含 system），使 system+history 不超预算，且保留最新一条消息。"""
        history = self.get_messages(session_id)
        while (
            len(history) > 1
            and self.count_tokens([system_message] + history) > self._token_budget
        ):
            history.pop(0)  # 丢最旧
        return history
