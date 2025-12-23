# Code Style Guide

## Overview

This project follows PEP 8 style guidelines with specific modifications for readability and consistency.

## Code Formatting

### Black

We use Black with a line length of 100 characters.

```bash
black src/ tests/ --line-length 100
```

### Import Sorting

We use isort with Black compatibility:

```bash
isort src/ tests/ --profile black
```

### Line Length

- Maximum line length: 100 characters
- Exceptions: URLs, long strings in tests

## Type Annotations

All functions must have type annotations:

```python
def predict_match(
    team1: List[int],
    team2: List[int],
    model: EnsembleModel
) -> PredictionResponse:
    """Predict match outcome."""
    pass
```

Use `typing` module for complex types:

```python
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray
```

## Docstrings

Use Google-style docstrings:

```python
def extract_features(
    hero_ids: List[int],
    embeddings: NDArray[np.float32]
) -> Dict[str, float]:
    """
    Extract features from hero selection.

    Args:
        hero_ids: List of hero IDs (length 5-6)
        embeddings: Hero embedding matrix

    Returns:
        Dictionary of extracted features

    Raises:
        ValueError: If team size is invalid

    Examples:
        >>> features = extract_features([1, 5, 10, 15, 20], embeddings)
        >>> print(features['tank_count'])
        0
    """
    pass
```

## Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_` (single underscore)
- **Module Private**: Prefix with `__` (double underscore)

## File Organization

1. Imports (stdlib, third-party, local)
2. Constants
3. Classes
4. Functions
5. `if __name__ == "__main__":`

```python
"""Module docstring."""

import os
from typing import List

import numpy as np
from fastapi import FastAPI

from src.utils.heroes import get_hero_metadata

CONSTANT_VALUE = 42

class MyClass:
    """Class docstring."""
    pass

def my_function() -> None:
    """Function docstring."""
    pass

if __name__ == "__main__":
    main()
```

## Error Handling

Use specific exceptions:

```python
if not validate_team(team):
    raise ValueError("Team composition is invalid")

try:
    model.load(path)
except FileNotFoundError:
    logger.error(f"Model file not found: {path}")
    raise
```

## Testing

- Test functions: `test_<function_name>`
- Test classes: `Test<ClassName>`
- Use descriptive test names
- One assertion per test when possible

```python
def test_extract_features_valid_input(
    feature_extractor: HeroFeatureExtractor
) -> None:
    """Test feature extraction with valid team composition."""
    team = [1, 5, 10, 15, 20]
    features = feature_extractor.extract_features(team)
    
    assert isinstance(features, dict)
    assert len(features) > 0
```

## Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

Hooks run:
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- bandit security scanning

## Linting

### flake8

```bash
flake8 src/ tests/ --max-line-length=100
```

Ignores:
- E203: Whitespace before ':'
- W503: Line break before binary operator

### mypy

```bash
mypy src/ --strict
```

## Best Practices

1. **Keep functions small**: Single responsibility
2. **Avoid deep nesting**: Maximum 3-4 levels
3. **Use meaningful names**: Self-documenting code
4. **Comment why, not what**: Code should explain itself
5. **DRY principle**: Don't repeat yourself
6. **Fail fast**: Validate inputs early
7. **Use type hints**: Help with IDE support and documentation

## Example

Good:

```python
def calculate_win_probability(
    team1: List[int],
    team2: List[int],
    model: EnsembleModel
) -> float:
    """
    Calculate win probability for team1.
    
    Args:
        team1: First team hero IDs
        team2: Second team hero IDs
        model: Trained ensemble model
        
    Returns:
        Win probability [0, 1]
        
    Raises:
        ValueError: If teams are invalid
    """
    validate_teams(team1, team2)
    features = extract_features(team1, team2)
    proba = model.predict_proba(features.reshape(1, -1))
    return float(proba[0, 1])
```

Bad:

```python
def calc(team1, team2, m):
    # Calculate probability
    f = extract(team1, team2)
    p = m.predict(f.reshape(1, -1))
    return p[0, 1]  # No validation, unclear what p is
```

