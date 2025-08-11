"""
I/O utilities for type-safe JSONL operations.

This module provides utilities for reading and writing JSONL files with Pydantic models,
following DataEnvGym patterns for robust data serialization.
"""

import json
from pathlib import Path
from typing import Type, TypeVar, Iterator, List, Union
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticJSONLinesWriter:
    """Writer for JSONL files with Pydantic model validation"""

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize writer.

        Args:
            file_path: Path to the output JSONL file
        """
        self.file_path = Path(file_path)

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, item: BaseModel) -> None:
        """
        Write a single Pydantic model to the file.

        Args:
            item: Pydantic model instance to write
        """
        with open(self.file_path, "a", encoding="utf-8") as f:
            json_str = item.model_dump_json()
            f.write(json_str + "\n")

    def write_batch(self, items: List[BaseModel]) -> None:
        """
        Write a batch of Pydantic models to the file.

        Args:
            items: List of Pydantic model instances to write
        """
        with open(self.file_path, "w", encoding="utf-8") as f:
            for item in items:
                json_str = item.model_dump_json()
                f.write(json_str + "\n")

    def append_batch(self, items: List[BaseModel]) -> None:
        """
        Append a batch of Pydantic models to the file.

        Args:
            items: List of Pydantic model instances to append
        """
        with open(self.file_path, "a", encoding="utf-8") as f:
            for item in items:
                json_str = item.model_dump_json()
                f.write(json_str + "\n")


class PydanticJSONLinesReader:
    """Reader for JSONL files with Pydantic model validation"""

    def __init__(self, file_path: Union[str, Path], model_class: Type[T]):
        """
        Initialize reader.

        Args:
            file_path: Path to the JSONL file to read
            model_class: Pydantic model class for parsing
        """
        self.file_path = Path(file_path)
        self.model_class = model_class

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over items in the file.

        Yields:
            Parsed Pydantic model instances
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    data = json.loads(line)
                    item = self.model_class.model_validate(data)
                    yield item
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to parse line {line_num} as {self.model_class.__name__}: {e}")

    def read_all(self) -> List[T]:
        """
        Read all items from the file.

        Returns:
            List of parsed Pydantic model instances
        """
        return list(self)

    def count_lines(self) -> int:
        """
        Count the number of non-empty lines in the file.

        Returns:
            Number of lines with content
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def write_jsonl(items: List[BaseModel], file_path: Union[str, Path]) -> None:
    """
    Convenience function to write a list of Pydantic models to JSONL.

    Args:
        items: List of Pydantic model instances
        file_path: Path to the output file
    """
    writer = PydanticJSONLinesWriter(file_path)
    writer.write_batch(items)


def read_jsonl(file_path: Union[str, Path], model_class: Type[T]) -> List[T]:
    """
    Convenience function to read a JSONL file into a list of Pydantic models.

    Args:
        file_path: Path to the JSONL file
        model_class: Pydantic model class for parsing

    Returns:
        List of parsed model instances
    """
    reader = PydanticJSONLinesReader(file_path, model_class)
    return reader.read_all()


def validate_jsonl_file(file_path: Union[str, Path], model_class: Type[T]) -> bool:
    """
    Validate that a JSONL file can be parsed with the given model class.

    Args:
        file_path: Path to the JSONL file
        model_class: Pydantic model class for validation

    Returns:
        True if valid, False otherwise
    """
    try:
        reader = PydanticJSONLinesReader(file_path, model_class)
        # Try to read all items to validate
        list(reader)
        return True
    except (FileNotFoundError, ValueError):
        return False
