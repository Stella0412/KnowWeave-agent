"""工具参数 JSON Schema 严格校验。

规则（与 spec 4 对齐）：
- 先校验 Schema 自身合法性，非法 Schema 抛 ValueError，不进入任何可调用清单。
- 参数必须是 JSON 对象（dict），即使 Schema 宽松或为空。
- 严格类型：字符串 "2" 不是 integer，布尔不是 integer，不做隐式转换。
- 收集全部错误，返回字段路径、原因，仅 enum 错误附带 allowed_values。
- 尊重声明的 $schema（draft-07 等）与本地 $ref。
- 不修改参数对象。
"""

from jsonschema import validators
from jsonschema.exceptions import SchemaError


def check_schema(schema: dict) -> None:
    """校验工具参数 Schema 自身合法性；非法时抛 ValueError。"""
    try:
        validators.validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid tool parameter schema: {exc.message}") from exc


def argument_errors(schema: dict, arguments: object) -> list[dict]:
    """严格校验参数，返回全部错误。Schema 非法时先抛 ValueError。"""
    check_schema(schema)

    if not isinstance(arguments, dict):
        return [{"path": [], "message": "arguments must be a JSON object"}]

    validator = validators.validator_for(schema)(schema)
    errors: list[dict] = []
    for error in validator.iter_errors(arguments):
        item: dict = {"path": list(error.path), "message": error.message}
        if error.validator == "enum":
            item["allowed_values"] = list(error.validator_value)
        errors.append(item)
    return errors
