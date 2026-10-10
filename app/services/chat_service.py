from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.db.repository import ChatRepository
from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.context_service import ContextStore
from app.tools.executor import execute_tool
from app.tools.registry import build_tools


def _row_to_message(row) -> BaseMessage:
    if row.role.value == "user":
        return HumanMessage(content=row.content or "")
    if row.role.value == "assistant":
        return AIMessage(content=row.content or "", tool_calls=row.tool_calls or [])
    return ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id or "")


class ChatService:
    def __init__(
        self,
        model,
        session_factory,
        token_budget: int = 4096,
        tool_timeout: float = 5.0,
        tool_retries: int = 1,
    ):
        self._model = model
        self._session_factory = session_factory
        self._store = ContextStore(token_budget)
        self._tool_timeout = tool_timeout
        self._tool_retries = tool_retries

    async def stream(self, session_id: str | None, message: str):
        try:
            # 1. 解析/创建会话，落 user 消息，读历史
            async with self._session_factory() as session:
                repo = ChatRepository(session)
                if session_id is None:
                    conversation_id = await repo.create_conversation()
                    await session.commit()
                    yield {"type": "session", "session_id": str(conversation_id)}
                else:
                    conversation_id = int(session_id)
                await repo.add_message(conversation_id, role="user", content=message)
                await session.commit()
                rows = await repo.list_messages(conversation_id)
            history = [_row_to_message(r) for r in rows]

            # 2. 裁剪 + 决策
            system = SystemMessage(content=SYSTEM_PROMPT)
            trimmed = self._store.trim(system, history)
            decision_messages = [system] + trimmed
            tools = build_tools(self._session_factory, conversation_id)
            ai_msg = await self._model.bind_tools(tools).ainvoke(decision_messages)

            if ai_msg.tool_calls:
                # 3a. 命中工具：推状态帧 → 落 assistant(tool_calls) → 执行 → 落 tool → 回灌
                for tc in ai_msg.tool_calls:
                    yield {"type": "tool_status", "name": tc["name"], "status": "running"}
                async with self._session_factory() as session:
                    repo = ChatRepository(session)
                    await repo.add_message(
                        conversation_id, role="assistant", content="", tool_calls=ai_msg.tool_calls
                    )
                    await session.commit()
                by_name = {t.name: t for t in tools}
                tool_msgs = [
                    await execute_tool(
                        tc, by_name, timeout=self._tool_timeout, retries=self._tool_retries
                    )
                    for tc in ai_msg.tool_calls
                ]
                async with self._session_factory() as session:
                    repo = ChatRepository(session)
                    for tm in tool_msgs:
                        await repo.add_message(
                            conversation_id, role="tool", content=tm.content,
                            tool_call_id=tm.tool_call_id,
                        )
                    await session.commit()
                final_messages = decision_messages + [ai_msg] + tool_msgs
            else:
                # 3b. 未命中工具：直接回答
                final_messages = decision_messages

            # 4. 最终回答流式（不绑工具，杜绝循环）
            collected: list[str] = []
            async for chunk in self._model.astream(final_messages):
                if chunk.content:
                    collected.append(chunk.content)
                    yield {"type": "token", "content": chunk.content}

            # 5. 落 assistant 全文
            async with self._session_factory() as session:
                repo = ChatRepository(session)
                await repo.add_message(
                    conversation_id, role="assistant", content="".join(collected)
                )
                await session.commit()
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
