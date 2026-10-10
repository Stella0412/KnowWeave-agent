from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.prompts.customer_service import SYSTEM_PROMPT

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
