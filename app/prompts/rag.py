"""ch04 在线管线用到的三组 Prompt：改写 / 生成前自评 / 忠实度裁判。

三个都走结构化输出（`with_structured_output(..., method="function_calling")`，
DeepSeek 不支持 json_schema 形式的 response_format，见 app/core/llm.py 的注释）。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class RewriteOutput(BaseModel):
    """一次调用双输出：给 dense 路的标准问法 + 给 BM25 路的同义词。"""

    standard_query: str = Field(description="改写后的一句标准问法，保留原问题里的具体信息（型号、数字、周期）")
    synonyms: list[str] = Field(
        default_factory=list,
        description="知识库文档里可能用的同义词/别称，用于字面检索；没有就给空数组",
    )


class SelfCheckOutput(BaseModel):
    sufficient: bool = Field(description="下面的资料是否足以回答用户的问题")
    reason: str = Field(description="判断依据，一句话；不足以回答时写明缺什么")


class FaithJudgeOutput(BaseModel):
    faithful: bool = Field(description="答案里的每一句是否都能被资料支持")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="没有资料支持的句子原文，逐句列出；没有就给空数组"
    )
    reason: str = Field(description="判定理由，指明编在哪一句")


REWRITE_SYSTEM = """你是电商客服知识库的检索助手。把用户的口语问法改写成便于检索的写法。

要求：
1. standard_query：一句标准问法，**必须保留原问题里的具体信息**（商品型号、数字、时间周期、
   专有名词），只把口语和指代替换掉；不要新增原问题里没有的诉求。
2. synonyms：知识库文档里可能出现的同义词、别称或书面说法，用于字面检索。例如
   「邮费」↔「运费」、「不想要了」↔「无理由退货」。最多 5 个，宁缺勿滥。
3. 不要回答问题，只做改写。"""


SELF_CHECK_SYSTEM = """你在判断「手上的资料够不够回答这个问题」。

只依据给定资料判断，不要用你自己的常识补充：
- 资料里能找到直接依据 → sufficient=true
- 资料里没有依据、只有沾边的只言片语、或需要外推猜测才能作答 → sufficient=false
- 资料里明确写着「平台不承诺」这类内容 → 那也算有依据（sufficient=true），
  该怎么措辞是回答阶段的事，不是这里要判的。

reason 用一句话说清判断依据；判 false 时写明缺哪部分信息。"""


FAITH_JUDGE_SYSTEM = """你在做忠实度审查：判断客服答案里的每一句是否有资料支持。

规则：
1. 逐句核对。只依据给定资料，不利用你自己的常识。
2. 资料里没有、且答案却写出来的具体信息（数字、时间、型号、政策承诺）→ 这是编造。
3. 引用编号写错（比如只给了 [1][2] 却标了 [3]）也算 unsupported。
4. 资料里的内容概括、换句话复述 → 算支持。
5. 拒答类答案（说明未查到依据、建议转人工）没有可判的断言 → faithful=true。

输出 unsupported_claims 时**逐句抄答案原文**，理由里指明编在哪一句，便于人工复核。"""


def rewrite_messages(raw_query: str) -> list:
    return [
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=f"用户原话：{raw_query}"),
    ]


def self_check_messages(raw_query: str, evidence_block: str) -> list:
    return [
        SystemMessage(content=SELF_CHECK_SYSTEM),
        HumanMessage(content=f"【用户的问题】\n{raw_query}\n\n【手上的资料】\n{evidence_block}"),
    ]


def faith_judge_messages(query: str, answer: str, evidence_block: str) -> list:
    return [
        SystemMessage(content=FAITH_JUDGE_SYSTEM),
        HumanMessage(
            content=f"【用户的问题】\n{query}\n\n【客服的回答】\n{answer}\n\n【本轮给模型的资料】\n{evidence_block}"
        ),
    ]
