from langchain_openai import ChatOpenAI

from app.config import Settings


def build_chat_model(
    settings: Settings,
    temperature: float = 0.7,
    max_retries: int = 8,
    *,
    model_name: str | None = None,
) -> ChatOpenAI:
    """统一 OpenAI 协议直连上游（DeepSeek / GPT / Ollama 换 base_url 即可）。

    生成侧保留 0.7 让回答自然；裁判/自评这类判定调用传 0，避免同一答案两次判出不同结论。
    max_retries 默认 8：300 题评估要跑几十分钟，一次 DNS 抖动不该让整轮作废。
    model_name 可选覆盖 settings.llm_model（ch06 意图识别按用途指定模型）。
    """
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model_name or settings.llm_model,
        temperature=temperature,
        streaming=True,
        max_retries=max_retries,
    )


def build_structured_model(
    settings: Settings,
    schema,
    temperature: float = 0.7,
    max_retries: int = 8,
    *,
    model_name: str | None = None,
):
    model = build_chat_model(
        settings, temperature=temperature, max_retries=max_retries,
        model_name=model_name,
    )
    # DeepSeek 不支持 json_schema 类型的 response_format（langchain-openai 默认 method），
    # 改用 function_calling（走 tools/tool_choice），GPT/Claude/Ollama 同样通用。见 SDD Ruling 4。
    return model.with_structured_output(schema, method="function_calling")
