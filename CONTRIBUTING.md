# Contributing Guidelines

Thank you for your interest in contributing to OverWatch Win Predictor!

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Format code: `make format`
7. Commit changes: `git commit -m "Add your feature"`
8. Push to branch: `git push origin feature/your-feature-name`
9. Open a Pull Request

## Development Setup

1. Clone your fork
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install dependencies: `make install-dev`
5. Install pre-commit hooks: `pre-commit install`

## Code Standards

- Follow PEP 8 style guide
- Use type annotations for all functions
- Write docstrings (Google style)
- Keep functions small and focused
- Write tests for new features
- Ensure test coverage stays above 80%

## Testing

- Write tests for all new features
- Run tests before committing: `pytest`
- Aim for 80%+ coverage
- Use descriptive test names

## Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Ensure code quality checks pass
4. Request review from maintainer

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add hero recommendation endpoint
fix: Fix validation error for duplicate heroes
docs: Update API documentation
test: Add tests for feature extraction
```

## Questions?

Feel free to open an issue for questions or discussion.

