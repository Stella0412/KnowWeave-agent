"""ch08 · 建单意图证据：只从落库用户消息文本推导。

授权唯一来源是「持久化后的用户原话」。update_ticket_request 是纯同步函数
（无 IO、无 DB、无模型调用）；模型输出、指代消解文本、工具参数、create_ticket
参数一律不能产生授权。

状态语义（active / cancelled / consumed）：
- active：用户以本人语气明确要求建单（帮我建个工单 / 帮我提个工单 / 给我建工单）。
- cancelled：明确撤回（不要建了 / 算了不建工单 / 不用建工单）；撤回后不可复用，
  之后新的明确请求开启一条新的 active 证据。
- consumed：只在工单真正写库成功后由上层置为 consumed；本模块永不产出 consumed，
  且 consumed 证据不会被后续补充文本重新提升为 active。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

QUOTE_MAX_CHARS = 200

# 建单动词 + 工单（. 不跨换行的少量修饰：个/一个/一下 等）
_TICKET = r"(建|提|开|创建|建立|提交).{0,4}工单"

# 第一人称祈使/意愿前缀 + 建单动词
_REQUEST = re.compile(
    r"(帮我|给我|为我|麻烦帮我|请帮我|麻烦给我|请给我|帮忙|我要|我想|我需要|"
    r"希望帮|希望给我|劳驾帮我)"
    r".{0,8}" + _TICKET
)
# 祈使句开头直接建单：「建个工单」
_BARE_REQUEST = re.compile(r"^" + _TICKET)

# 引用 / 第三人称转述标记
_QUOTE_CHAR = re.compile(r"[‘’“”「」『』\"']")
_THIRD_PARTY = re.compile(
    r"(他说|她说|他们说|朋友说|朋友让|别人说|别人让|"
    r"让(我|他|她|他们|人家)|叫(我|他|她|人家)|人家说|同事说|家人说|转述)"
)

# 条件句
_CONDITIONAL = re.compile(r"(如果|假如|要是|倘若|若|万一|一旦|的话).{0,12}" + _TICKET)

# 疑问句（询问怎么建单 / 确认含义等，不是授权）
_QUESTION = re.compile(r"(怎么|如何|为什么|什么|是什么意思|怎么样|干嘛|为啥|是怎样|吗|呢)")

# 否定 / 撤回：明确「不建」「别建」「取消建单」等
_NEGATED = re.compile(r"(不要|不用|别|不需要|不必|无需|别想|休想).{0,6}" + _TICKET)
_WITHDRAWAL = re.compile(
    r"(算了|要不|那)?(不要建|不用建|别建|不建)(工单|单|了)?"
    r"|(取消|撤销|撤回)(建单|工单|单)"
)


@dataclass
class TicketRequestEvidence:
    """建单请求证据（字段顺序固定：message_id / quote / status）。"""

    message_id: int
    quote: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _coerce(previous):
    if previous is None:
        return None
    if isinstance(previous, TicketRequestEvidence):
        return previous
    if isinstance(previous, dict):
        try:
            return TicketRequestEvidence(
                message_id=previous.get("message_id"),
                quote=previous.get("quote") or "",
                status=previous.get("status") or "",
            )
        except (TypeError, ValueError):
            return None
    return None


def _excerpt(text: str, match: re.Match) -> str:
    return text[match.start():match.end()][:QUOTE_MAX_CHARS]


def _explicit_request_match(text: str) -> re.Match | None:
    match = _REQUEST.search(text)
    if match is not None:
        return match
    return _BARE_REQUEST.search(text)


def _is_explicit_request(text: str) -> bool:
    if _explicit_request_match(text) is None:
        return False
    # 排除：引用他人原话 / 第三人称转述 / 条件句 / 疑问句
    if _QUOTE_CHAR.search(text):
        return False
    if _THIRD_PARTY.search(text):
        return False
    if _CONDITIONAL.search(text):
        return False
    if _QUESTION.search(text):
        return False
    return True


def update_ticket_request(previous, message_id, user_text):
    """由落库用户消息文本推导建单证据。

    previous：TicketRequestEvidence | dict | None（上一轮 checkpoint 里的证据）。
    message_id：本轮持久化后的用户消息 id。
    user_text：本轮持久化后的用户原话。

    返回 TicketRequestEvidence | None；「不确定」一律返回 None，宁可让流程再次
    追问明确请求，也不把模糊文本当成授权。
    """
    text = _normalize(user_text)
    if not text or not message_id:
        return None

    prev = _coerce(previous)
    prev_active = prev if (prev is not None and prev.status == "active") else None

    # 1. 否定 / 撤回：有 active 前值则置 cancelled；否则（孤立否定）不产生证据。
    withdrawal = _WITHDRAWAL.search(text)
    if withdrawal is None:
        withdrawal = _NEGATED.search(text)
    if withdrawal is not None:
        if prev_active is not None:
            return TicketRequestEvidence(
                message_id=message_id,
                quote=_excerpt(text, withdrawal),
                status="cancelled",
            )
        return None

    # 2. 明确请求 → 新的 active 证据（consumed/cancelled 不会被补充文本复用，
    #    但新的明确请求总是开启新证据）。
    request_match = _explicit_request_match(text)
    if request_match is not None and _is_explicit_request(text):
        return TicketRequestEvidence(
            message_id=message_id,
            quote=_excerpt(text, request_match),
            status="active",
        )

    # 3. 补充描述 / 一般文本：保留既有 active 证据（message_id 与 quote 不变）。
    if prev_active is not None:
        return prev_active

    return None
