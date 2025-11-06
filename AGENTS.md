# Python Project Agent Guide

## Environment
This project uses **uv** for package management.
To install dependencies, always use:
```bash
uv add <package>
```

The environment should include:
- Python 3.12+
- ruff (linting / formatting)
- pytest (testing)
- typing-extensions, numpy, pandas, scipy (common base)

## Code Style
- Follow **PEP8** and **PEP484** (type hints are required).
- Use `ruff` for linting and formatting:
  ```bash
  ruff check .
  ruff format .
  ```
- Avoid wildcard imports (`from module import *`).
- Use explicit imports for clarity.
- Maximum line length: 100 characters.

### Type Hints

- Always use **built-in generics** (PEP 585) instead of legacy `typing` imports.
  ```python
  # ✅ Recommended
  def process(data: list[int], mapping: dict[str, float]) -> None:
      ...

  # 🚫 Avoid
  from typing import List, Dict
  def process(data: List[int], mapping: Dict[str, float]) -> None:
      ...
  ```

- Use `|` (PEP 604) instead of `Union`:
  ```python
  # ✅ Recommended
  def load(path: str | Path) -> bytes: ...

  # 🚫 Avoid
  from typing import Union
  def load(path: Union[str, Path]) -> bytes: ...
  ```

- Use `Optional` only when necessary, otherwise prefer `| None`.
  ```python
  # ✅ Recommended
  def normalize(x: np.ndarray | None = None) -> np.ndarray: ...
  ```

- Type hints should always describe *structure*, not *intent*.
  Prefer `Mapping[str, Any]` over `dict` when immutability or interface clarity matters.

## Testing
- Unit tests live under the `tests/` directory.
- Use `pytest` with short test names and clear assertions.
- Prefer pure functions and deterministic results.

Example test:
```python
def test_add():
    assert add(1, 2) == 3
```

## Project Structure
```
src/
  module1/
    __init__.py
  module2/
tests/
pyproject.toml
```

## Documentation
- Use **Google-style docstrings** with examples and shapes.
- Each function should describe arguments, return types, and complexity.

Example:
```python
def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to zero mean and unit variance.

    Args:
        x (np.ndarray): Input array of shape (n,).

    Returns:
        np.ndarray: Normalized array of same shape.

    Example:
        >>> normalize(np.array([1, 2, 3]))
        array([-1.22, 0.0, 1.22])
    """
```

## Coding Preferences
- Use dataclasses for structured data.
- Avoid hard-coded paths; prefer environment variables.
- Prefer `pathlib` over `os.path`.
- Use `logging` instead of `print`.

## Expected Tasks for Codex
Codex should:
- Generate Python modules following the above conventions.
- Create tests automatically for new functions.
- Suggest ruff-compliant code and formatting.
- Use `uv add` for dependency additions.
- Respect type hints and use numpy/scipy idioms for numeric code.

## Markdown Documents

- Prefer markdown file as documents
- Use latex and $ for math
- Mermaid charts are better than ascii
