.PHONY: help install install-dev test lint format security clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run all linters"
	@echo "  make format        - Format code with black and isort"
	@echo "  make security      - Run security checks"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make clean         - Clean cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

security:
	@mkdir -p reports
	bandit -r src/ -f json -o reports/bandit-report.json || true
	safety check --file requirements.txt || true
	@echo "Security reports generated in reports/"

docker-build:
	docker build -t overwatch-predictor:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ dist/ build/ .eggs *.egg-info
	rm -rf reports/

