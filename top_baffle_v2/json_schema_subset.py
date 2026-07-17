"""Deterministic pure-stdlib validator for the release-catalog schema.

This is intentionally a small JSON Schema 2020-12 subset, not a permissive
approximation and not a general replacement for ``jsonschema``.  The release
consumer owns one checked-in schema and fails closed if that schema starts
using a validation keyword this module does not implement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any


class JsonSchemaSubsetError(ValueError):
    """Raised for an invalid instance or an unsupported/malformed schema."""


_ANNOTATION_KEYWORDS = {"$schema", "$id", "title"}
_VALIDATION_KEYWORDS = {
    "$ref", "$defs", "type", "const", "enum", "required",
    "properties", "additionalProperties", "minLength", "pattern",
    "minItems", "maxItems", "items", "prefixItems",
    "exclusiveMinimum", "anyOf", "allOf", "if", "then",
}
_SUPPORTED_KEYWORDS = _ANNOTATION_KEYWORDS | _VALIDATION_KEYWORDS
_JSON_TYPES = {
    "null", "boolean", "object", "array", "number", "integer", "string",
}


def _instance_path(path: str, key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _schema_error(path: str, message: str) -> JsonSchemaSubsetError:
    return JsonSchemaSubsetError(f"schema {path}: {message}")


def _preflight(schema: Any, path: str = "#") -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping):
        raise _schema_error(path, "schema node must be an object or boolean")
    unsupported = sorted(set(schema) - _SUPPORTED_KEYWORDS)
    if unsupported:
        raise _schema_error(
            path, "unsupported keyword(s): " + ", ".join(unsupported))

    reference = schema.get("$ref")
    if reference is not None and not isinstance(reference, str):
        raise _schema_error(f"{path}/$ref", "must be a string")

    type_value = schema.get("type")
    if type_value is not None:
        declared = ([type_value] if isinstance(type_value, str)
                    else type_value)
        if (not isinstance(declared, Sequence)
                or isinstance(declared, (str, bytes))
                or not declared
                or any(item not in _JSON_TYPES for item in declared)):
            raise _schema_error(f"{path}/type", "has unsupported JSON type")

    required = schema.get("required")
    if required is not None and (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)):
        raise _schema_error(f"{path}/required", "must be an array of strings")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise _schema_error(f"{path}/enum", "must be a non-empty array")

    for keyword in ("minLength", "minItems", "maxItems"):
        value = schema.get(keyword)
        if value is not None and (
                type(value) is not int or value < 0):
            raise _schema_error(
                f"{path}/{keyword}", "must be a nonnegative integer")
    minimum = schema.get("exclusiveMinimum")
    if minimum is not None and (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(float(minimum))):
        raise _schema_error(
            f"{path}/exclusiveMinimum", "must be a finite number")
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise _schema_error(f"{path}/pattern", "must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise _schema_error(
                f"{path}/pattern", f"invalid regular expression: {exc}") from exc

    for keyword in ("$defs", "properties"):
        children = schema.get(keyword)
        if children is None:
            continue
        if (not isinstance(children, Mapping)
                or any(not isinstance(key, str) for key in children)):
            raise _schema_error(f"{path}/{keyword}", "must be an object")
        for key in sorted(children):
            _preflight(children[key], f"{path}/{keyword}/{key}")

    additional = schema.get("additionalProperties")
    if additional is not None:
        if not isinstance(additional, (bool, Mapping)):
            raise _schema_error(
                f"{path}/additionalProperties",
                "must be a boolean or schema")
        if isinstance(additional, Mapping):
            _preflight(additional, f"{path}/additionalProperties")

    items = schema.get("items")
    if items is not None:
        _preflight(items, f"{path}/items")
    prefix_items = schema.get("prefixItems")
    if prefix_items is not None:
        if not isinstance(prefix_items, list):
            raise _schema_error(f"{path}/prefixItems", "must be an array")
        for index, child in enumerate(prefix_items):
            _preflight(child, f"{path}/prefixItems/{index}")

    for keyword in ("anyOf", "allOf"):
        children = schema.get(keyword)
        if children is None:
            continue
        if not isinstance(children, list) or not children:
            raise _schema_error(f"{path}/{keyword}", "must be non-empty")
        for index, child in enumerate(children):
            _preflight(child, f"{path}/{keyword}/{index}")
    for keyword in ("if", "then"):
        child = schema.get(keyword)
        if child is not None:
            _preflight(child, f"{path}/{keyword}")


def _resolve_reference(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise JsonSchemaSubsetError(
            f"schema #/$ref: only local JSON Pointer references are supported; "
            f"got {reference!r}")
    current: Any = root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                raise JsonSchemaSubsetError(
                    f"schema #/$ref: unresolved reference {reference!r}")
            current = current[index]
        else:
            raise JsonSchemaSubsetError(
                f"schema #/$ref: unresolved reference {reference!r}")
    return current


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return (len(left) == len(right)
                and all(_json_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, Mapping):
        return (set(left) == set(right)
                and all(_json_equal(left[key], right[key]) for key in left))
    return left == right


def _matches_type(instance: Any, declared: str) -> bool:
    if declared == "null":
        return instance is None
    if declared == "boolean":
        return type(instance) is bool
    if declared == "object":
        return isinstance(instance, Mapping)
    if declared == "array":
        return isinstance(instance, list)
    if declared == "string":
        return isinstance(instance, str)
    if declared == "number":
        return (not isinstance(instance, bool)
                and isinstance(instance, (int, float))
                and math.isfinite(float(instance)))
    if declared == "integer":
        return (not isinstance(instance, bool)
                and isinstance(instance, (int, float))
                and math.isfinite(float(instance))
                and float(instance).is_integer())
    return False


def _validate(
    instance: Any,
    schema: Any,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if schema is True:
        return
    if schema is False:
        raise JsonSchemaSubsetError(f"{path}: rejected by false schema")
    assert isinstance(schema, Mapping)

    reference = schema.get("$ref")
    if reference is not None:
        _validate(instance, _resolve_reference(root, reference), root, path)

    type_value = schema.get("type")
    if type_value is not None:
        declared = [type_value] if isinstance(type_value, str) else type_value
        if not any(_matches_type(instance, item) for item in declared):
            expected = ", ".join(declared)
            raise JsonSchemaSubsetError(
                f"{path}: expected type {expected}, got "
                f"{type(instance).__name__}")

    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise JsonSchemaSubsetError(
            f"{path}: value does not equal const {schema['const']!r}")
    if "enum" in schema and not any(
            _json_equal(instance, value) for value in schema["enum"]):
        raise JsonSchemaSubsetError(f"{path}: value is not in enum")

    if isinstance(instance, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(instance) < minimum:
            raise JsonSchemaSubsetError(
                f"{path}: string length {len(instance)} is below "
                f"minLength {minimum}")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            raise JsonSchemaSubsetError(
                f"{path}: string does not match pattern {pattern!r}")

    if isinstance(instance, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(instance) < minimum:
            raise JsonSchemaSubsetError(
                f"{path}: item count {len(instance)} is below "
                f"minItems {minimum}")
        if maximum is not None and len(instance) > maximum:
            raise JsonSchemaSubsetError(
                f"{path}: item count {len(instance)} exceeds "
                f"maxItems {maximum}")
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(prefix[:len(instance)]):
            _validate(instance[index], child, root, f"{path}[{index}]")
        if "items" in schema:
            for index in range(len(prefix), len(instance)):
                _validate(
                    instance[index], schema["items"], root,
                    f"{path}[{index}]")

    if isinstance(instance, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise JsonSchemaSubsetError(
                f"{path}: missing required propert"
                f"{'y' if len(missing) == 1 else 'ies'} "
                + ", ".join(repr(key) for key in missing))
        properties = schema.get("properties", {})
        for key in sorted(set(instance) & set(properties)):
            _validate(
                instance[key], properties[key], root,
                _instance_path(path, key))
        additional = schema.get("additionalProperties", True)
        extras = sorted(set(instance) - set(properties))
        if additional is False and extras:
            raise JsonSchemaSubsetError(
                f"{path}: additional properties are forbidden: "
                + ", ".join(repr(key) for key in extras))
        if isinstance(additional, Mapping):
            for key in extras:
                _validate(
                    instance[key], additional, root,
                    _instance_path(path, key))

    exclusive_minimum = schema.get("exclusiveMinimum")
    if (exclusive_minimum is not None
            and _matches_type(instance, "number")
            and not float(instance) > float(exclusive_minimum)):
        raise JsonSchemaSubsetError(
            f"{path}: number must be greater than "
            f"{exclusive_minimum!r}")

    for index, child in enumerate(schema.get("allOf", [])):
        try:
            _validate(instance, child, root, path)
        except JsonSchemaSubsetError as exc:
            raise JsonSchemaSubsetError(
                f"{path}: allOf branch {index} failed: {exc}") from exc

    any_of = schema.get("anyOf")
    if any_of is not None:
        failures = []
        for child in any_of:
            try:
                _validate(instance, child, root, path)
            except JsonSchemaSubsetError as exc:
                failures.append(str(exc))
            else:
                break
        else:
            detail = failures[0] if failures else "no branch"
            raise JsonSchemaSubsetError(
                f"{path}: no anyOf branch matched; first failure: {detail}")

    condition = schema.get("if")
    if condition is not None and _schema_matches(instance, condition, root, path):
        then = schema.get("then")
        if then is not None:
            try:
                _validate(instance, then, root, path)
            except JsonSchemaSubsetError as exc:
                raise JsonSchemaSubsetError(
                    f"{path}: conditional then failed: {exc}") from exc


def _schema_matches(
    instance: Any,
    schema: Any,
    root: Mapping[str, Any],
    path: str,
) -> bool:
    try:
        _validate(instance, schema, root, path)
    except JsonSchemaSubsetError:
        return False
    return True


def validate_json_schema(instance: Any, schema: Any) -> None:
    """Validate ``instance`` or raise :class:`JsonSchemaSubsetError`."""
    if not isinstance(schema, Mapping):
        raise JsonSchemaSubsetError("schema #: root must be an object")
    _preflight(schema)
    _validate(instance, schema, schema, "$")
