"""Optional Pydantic design-document integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from ..contracts import DesignSpec, InputMigrator
from ..errors import ConfigurationError


TModel = TypeVar("TModel")


class PydanticDesignSpec(DesignSpec[TModel], Generic[TModel]):
    """Build a core DesignSpec around a Pydantic v2 model type."""

    def __init__(
        self,
        *,
        name: str,
        model_type: type[TModel],
        schema: Mapping[str, Any] | None = None,
        input_migrator: InputMigrator | None = None,
        normalizer: Callable[[Any], TModel] | None = None,
    ) -> None:
        model_validate = getattr(model_type, "model_validate", None)
        model_json_schema = getattr(model_type, "model_json_schema", None)
        if not callable(model_validate) or not callable(model_json_schema):
            raise ConfigurationError("PydanticDesignSpec requires a Pydantic v2 BaseModel subclass.")

        def validate(value: Any) -> TModel:
            return model_validate(value, strict=True)

        def normalize(value: Any) -> TModel:
            if isinstance(value, model_type):
                copy_method = getattr(value, "model_copy")
                return copy_method(deep=True)
            return model_validate(value)

        def serialize(value: TModel) -> Mapping[str, Any]:
            dump = getattr(value, "model_dump", None)
            if not callable(dump):
                raise TypeError("Pydantic design values must define model_dump().")
            return dump(mode="json", round_trip=True)

        super().__init__(
            name=name,
            schema=dict(schema) if schema is not None else model_json_schema(),
            validator=validate,
            serializer=serialize,
            normalizer=normalizer or normalize,
            input_migrator=input_migrator,
        )


__all__ = ["PydanticDesignSpec"]
