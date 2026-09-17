"""Strict, deterministic JSON records without runtime dependencies."""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T", bound="JsonRecord")


def _decode(annotation: object, value: object) -> object:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(option, value)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"value does not match {annotation}")
    if origin is Literal:
        if value not in args or not any(type(value) is type(a) for a in args):
            raise ValueError(f"expected one of {args}")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("expected an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v) for v in value)
        if len(value) != len(args):
            raise ValueError("array length differs from contract")
        return tuple(_decode(a, v) for a, v in zip(args, value, strict=True))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, JsonRecord):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise ValueError("expected an object")
        hints = get_type_hints(annotation)
        if not is_dataclass(annotation):
            raise TypeError("JSON records must be dataclasses")
        names = {f.name for f in fields(annotation)}
        if set(value) != names:
            raise ValueError(
                f"fields differ: missing={names - set(value)}, extra={set(value) - names}"
            )
        decoded = {k: _decode(hints[k], v) for k, v in value.items()}
        return cast(Callable[..., object], annotation)(**decoded)
    if annotation is float and type(value) in (int, float):
        converted = float(cast("int | float", value))
        if not isfinite(converted):
            raise ValueError("numbers must be finite")
        return converted
    if annotation in (str, int, bool, type(None)) and type(value) is annotation:
        if isinstance(value, str) and not value.strip():
            raise ValueError("strings must not be empty")
        return value
    raise ValueError(f"expected {annotation}")


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, JsonRecord) and is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


class JsonRecord:
    """Base for immutable records; reject unknown fields and implicit type coercion."""

    def __post_init__(self) -> None:
        for name, annotation in get_type_hints(type(self)).items():
            value = getattr(self, name)
            decoded = _decode(annotation, value)
            object.__setattr__(self, name, decoded)

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _encode(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls: type[T], value: object) -> T:
        return cast(T, _decode(cls, value))

    @classmethod
    def from_json(cls: type[T], value: str) -> T:
        return cls.from_dict(load_json(value))


def load_json(value: str) -> object:
    """Reject duplicate keys and non-finite constants for all platform envelopes."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def finite_float(value: str) -> float:
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("JSON numbers must be finite")
        return converted

    return json.loads(
        value, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float
    )
