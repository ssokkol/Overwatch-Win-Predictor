# OverWatch Win Predictor

An end-to-end ML service for analyzing team compositions in Overwatch 2, predicting match outcomes, and recommending optimal hero picks/counter-picks.

## Overview

The OverWatch Win Predictor is a production-ready machine learning system that analyzes hero selections from both teams and predicts the probability of victory with 80-85% accuracy. It uses data from competitive matches and provides real-time recommendations for optimal team compositions.

## Features

- **Match Outcome Prediction**: Predicts win probability for team compositions using ensemble ML models
- **Hero Recommendations**: Suggests optimal hero picks based on current and enemy team compositions
- **Feature Engineering**: Advanced feature extraction including Hero2Vec embeddings and team composition analysis
- **Production-Ready API**: FastAPI-based REST API with security, rate limiting, and comprehensive validation
- **MLOps Integration**: MLflow for experiment tracking and DVC for data versioning
- **Comprehensive Testing**: 80%+ test coverage with unit, integration, and security tests

## Technology Stack

### Core ML Stack
- Python 3.10+
- scikit-learn, XGBoost, PyTorch
- pandas, numpy, scipy
- category_encoders

### API & Deployment
- FastAPI + Uvicorn
- Redis (caching, rate limiting)
- Docker + Docker Compose

### MLOps
- MLflow (experiment tracking)
- DVC (data versioning, pipelines)

### Code Quality & Security
- black, isort, flake8, mypy
- bandit, safety (security scanning)
- pre-commit hooks
- GitHub Actions CI/CD

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional)
- Redis (optional, for production)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/overwatch-win-predictor.git
cd overwatch-win-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Install pre-commit hooks:
```bash
pre-commit install
```

### Running the Application

#### Development Mode

1. Start Redis (optional):
```bash
docker-compose up -d redis
```

2. Generate synthetic training data and train models:
```bash
python -m src.data.generate_synthetic
# Train models (see training scripts)
```

3. Run the API server:
```bash
uvicorn src.api.app:app --reload --port 8000
```

4. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. For production:
```bash
docker-compose -f docker-compose.prod.yml up --build
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Match Outcome

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "team1": {"hero_ids": [1, 5, 10, 15, 20]},
    "team2": {"hero_ids": [2, 7, 12, 17, 22]}
  }'
```

### Get Hero Recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "current_team": {"hero_ids": [1, 5, 10, 15, 20]},
    "enemy_team": {"hero_ids": [2, 7, 12, 17, 22]},
    "num_recommendations": 5
  }'
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m security
```

### Code Quality

```bash
# Format code
make format
# or
black src/ tests/
isort src/ tests/

# Lint
make lint
# or
flake8 src/ tests/
mypy src/

# Security scan
make security
# or
bandit -r src/
safety check --file requirements.txt
```

### Makefile Commands

```bash
make install       # Install production dependencies
make install-dev   # Install development dependencies
make test          # Run tests with coverage
make lint          # Run all linters
make format        # Format code
make security      # Run security checks
make docker-build  # Build Docker image
make clean         # Clean cache files
```

## Project Structure

```
overwatch-win-predictor/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── data/              # Data collection and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── configs/               # Configuration files
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage
└── models/                # Trained models
```

## Documentation

- [API Documentation](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Security Policy](docs/SECURITY.md)
- [Code Style Guide](docs/CODE_STYLE.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Model Training

See the training notebooks in `notebooks/` for step-by-step model training:

1. `01_data_collection.ipynb` - Data collection
2. `02_eda_heroes.ipynb` - Exploratory data analysis
3. `03_hero_embeddings.ipynb` - Hero2Vec training
4. `04_composition_analysis.ipynb` - Team composition analysis
5. `05_model_training.ipynb` - Model training and evaluation

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Acknowledgments

- Overwatch 2 game data and hero metadata
- MLflow for experiment tracking
- FastAPI for the web framework

