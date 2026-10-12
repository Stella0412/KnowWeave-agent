"""文本归一化去重（纯函数）。dense 向量本章只服务检索，去重走文本（spec §7 用户定）。"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """NFKC（全角转半角）+ 小写 + 去空白/标点/符号。"""
    folded = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", folded, flags=re.UNICODE)


def qa_fingerprint(question: str, answer: str) -> str:
    """指纹 = 归一化「问法 + 正文」。理由见 spec §7/§14：只按问法查会误杀共用节标题的多块。"""
    return f"{normalize_text(question)}\n{normalize_text(answer)}"
