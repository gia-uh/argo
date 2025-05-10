from pydantic import BaseModel
from typing import get_type_hints, Optional, Union


def type_to_str(tp):
    """Convert a type to a string representation."""
    # Handle Optional types (Union[..., None])
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", ())

    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]

        if len(non_none) == 1:
            return f"Optional[{type_to_str(non_none[0])}]"

    if hasattr(tp, "__name__"):
        return tp.__name__

    if hasattr(tp, "_name"):  # for generic types like List, Dict
        args_str = ", ".join(type_to_str(a) for a in args)
        return f"{tp._name}[{args_str}]"

    return str(tp).replace("typing.", "")


def generate_pydantic_code(model_cls: type[BaseModel]) -> str:
    """
    Generate Python source code for a Pydantic BaseModel subclass.
    """
    lines = []
    visited = set()

    def generate(cls, lines: list[str], visited: str):
        class_name = model_cls.__name__

        if cls.__name__ in visited:
            return

        visited.add(cls.__name__)

        # Generate the class definition
        lines.append(f"class {class_name}(BaseModel):")

        subtypes = []

        # Get type hints (annotations) for the model
        hints = get_type_hints(model_cls)

        # For each field, get name, type, and default value if any
        for field_name, field in model_cls.model_fields.items():
            field_type = hints.get(field_name, "Any")
            type_str = type_to_str(field_type)

            # Determine default value
            if not field.is_required:
                default_val = repr(field.default)
                line = f"    {field_name}: {type_str} = {default_val}"
            else:
                line = f"    {field_name}: {type_str}"

            lines.append(line)

            # if its a BaseModel, add to visited
            if issubclass(field_type, BaseModel):
                subtypes.append(field_type)

        if subtypes:
            lines.append("")

        for cls in subtypes:
            generate(cls, lines, visited)

    generate(model_cls, lines, visited)

    return "\n".join(lines)
