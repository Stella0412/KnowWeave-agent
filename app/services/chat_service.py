from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.context_service import ContextStore
from app.services.prompt_service import CHAT_PROMPT


class ChatService:
    def __init__(self, model, store: ContextStore):
        self._model = model
        self._store = store

    async def stream(self, session_id: str | None, message: str):
        try:
            if session_id is None:
                session_id = uuid4().hex
                yield {"type": "session", "session_id": session_id}
            system = SystemMessage(content=SYSTEM_PROMPT)
            history = self._store.trim_history(session_id, system)
            messages = CHAT_PROMPT.invoke(
                {"history": history, "input": message}
            ).to_messages()
            collected: list[str] = []
            async for chunk in self._model.astream(messages):
                content = chunk.content
                if content:
                    collected.append(content)
                    yield {"type": "token", "content": content}
            self._store.append(session_id, HumanMessage(content=message))
            self._store.append(session_id, AIMessage(content="".join(collected)))
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
