"""补齐本地 Langfuse 环境变量；已有值绝不覆盖，值绝不打印。"""

import secrets
import sys
from pathlib import Path


def _secret(prefix: str) -> str:
    return prefix + secrets.token_urlsafe(32)


def _hex32() -> str:
    return secrets.token_hex(32)


def _password() -> str:
    return secrets.token_urlsafe(18)


def _defaults() -> dict[str, str]:
    return {
        "LANGFUSE_BASE_URL": "http://localhost:3000",
        "LANGFUSE_TRACING_ENABLED": "true",
        "LANGFUSE_PROJECT_ID": "knowweave",
        "LANGFUSE_PUBLIC_KEY": _secret("pk-lf-local-"),
        "LANGFUSE_SECRET_KEY": _secret("sk-lf-local-"),
        "LANGFUSE_NEXTAUTH_SECRET": _hex32(),
        "LANGFUSE_SALT": _hex32(),
        "LANGFUSE_ENCRYPTION_KEY": _hex32(),
        "LANGFUSE_INIT_ORG_ID": "knowweave",
        "LANGFUSE_INIT_ORG_NAME": "知序",
        "LANGFUSE_INIT_PROJECT_ID": "knowweave",
        "LANGFUSE_INIT_PROJECT_NAME": "知序",
        "LANGFUSE_INIT_USER_EMAIL": "admin@knowweave.local",
        "LANGFUSE_INIT_USER_NAME": "admin",
        "LANGFUSE_INIT_USER_PASSWORD": _password(),
    }


def _is_placeholder(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return (
        (text.startswith("<") and text.endswith(">"))
        or lowered.startswith("your-")
        or lowered in {"changeme", "change-me"}
    )


def ensure_langfuse_env(path: Path) -> int:
    path = Path(path)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = original.splitlines()
    wanted = _defaults()
    seen: dict[str, int] = {}
    changed = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in wanted and key not in seen:
            seen[key] = index
            if _is_placeholder(value):
                lines[index] = f"{key}={wanted[key]}"
                changed += 1

    for key, value in wanted.items():
        if key not in seen:
            lines.append(f"{key}={value}")
            changed += 1

    output = "\n".join(lines)
    if output and not output.endswith("\n"):
        output += "\n"
    path.write_text(output, encoding="utf-8")
    return changed


def main(path: Path | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    changed = ensure_langfuse_env(Path(path))
    print(f"已补齐 {changed} 项，值未打印；Langfuse 登录邮箱见 .env")


if __name__ == "__main__":
    main()
