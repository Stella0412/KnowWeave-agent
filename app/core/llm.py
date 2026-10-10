from langchain_openai import ChatOpenAI

from app.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """统一 OpenAI 协议直连上游（DeepSeek / GPT / Ollama 换 base_url 即可）。"""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.7,
        streaming=True,
    )


def build_structured_model(settings: Settings, schema):
    model = build_chat_model(settings)
    return model.with_structured_output(schema)
