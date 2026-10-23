import pytest


SCHEMA = {
    "type": "object", "properties": {
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "state": {"type": "string", "enum": ["在保", "过保"]},
        "label": {"type": "string", "minLength": 2, "maxLength": 4},
    }, "required": ["count"], "additionalProperties": False,
}


def test_numeric_string_is_not_integer():
    from app.tools.validation import argument_errors

    assert argument_errors(SCHEMA, {"count": "2"})
    assert argument_errors(SCHEMA, {"count": 0})
    assert not argument_errors(SCHEMA, {"count": 2})


@pytest.mark.parametrize("arguments", [
    {}, {"count": True}, {"count": 6}, {"count": 1, "state": "未知"},
    {"count": 1, "label": "短"}, {"count": 1, "label": "超过四个字"},
    {"count": 1, "extra": "不允许"}, [], None, "{}", 2,
])
def test_invalid_tool_arguments_are_rejected(arguments):
    from app.tools.validation import argument_errors

    assert argument_errors(SCHEMA, arguments)


def test_arguments_must_be_object_even_with_permissive_schema():
    from app.tools.validation import argument_errors

    assert argument_errors({}, [])
    assert not argument_errors({}, {})


def test_errors_include_nested_path_reason_and_enum_choices():
    from app.tools.validation import argument_errors

    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"state": {"enum": ["在保", "过保"]}}
    }}}}
    errors = argument_errors(schema, {"items": [{"state": "未知"}]})
    assert len(errors) == 1
    assert errors[0]["path"] == ["items", 0, "state"]
    assert errors[0]["message"]
    assert errors[0]["allowed_values"] == ["在保", "过保"]
    assert set(errors[0]) == {"path", "message", "allowed_values"}


def test_collects_all_errors_without_mutating_arguments():
    from app.tools.validation import argument_errors

    arguments = {"count": "2", "state": "未知"}
    assert len(argument_errors(SCHEMA, arguments)) == 2
    assert arguments == {"count": "2", "state": "未知"}


def test_declared_draft_is_respected_and_local_refs_work():
    from app.tools.validation import argument_errors

    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
              "definitions": {"counts": {"type": "array", "items": [{"type": "integer"}]}},
              "properties": {"counts": {"$ref": "#/definitions/counts"}}}
    assert not argument_errors(schema, {"counts": [1]})
    assert argument_errors(schema, {"counts": ["1"]})


def test_invalid_schema_fails_before_validating_arguments():
    from app.tools.validation import argument_errors

    with pytest.raises(ValueError):
        argument_errors({"type": "invalid"}, {})
