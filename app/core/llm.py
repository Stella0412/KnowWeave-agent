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
    # DeepSeek 不支持 json_schema 类型的 response_format（langchain-openai 默认 method），
    # 改用 function_calling（走 tools/tool_choice），GPT/Claude/Ollama 同样通用。见 SDD Ruling 4。
    return model.with_structured_output(schema, method="function_calling")
