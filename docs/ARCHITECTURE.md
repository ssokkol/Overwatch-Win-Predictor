# System Architecture

## Overview

The OverWatch Win Predictor is built as a microservices-ready ML system with clear separation of concerns between data processing, feature engineering, model training, and API serving.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│              (Web Browser, Mobile App, etc.)                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Schemas &   │  │ Middleware   │  │  Rate        │     │
│  │ Validation   │  │ (Security,   │  │  Limiting    │     │
│  │              │  │  CORS)       │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Hero         │  │ Team         │  │ Hero2Vec     │     │
│  │ Features     │  │ Composition  │  │ Embeddings   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  XGBoost     │  │  Neural      │  │  Ensemble    │     │
│  │  Model       │  │  Network     │  │  Model       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Raw Data    │  │  Processed   │  │  Embeddings  │     │
│  │  (CSV)       │  │  Data        │  │  (numpy)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. API Layer

**FastAPI Application** (`src/api/app.py`)
- RESTful API endpoints
- Request/response validation with Pydantic
- Authentication and authorization
- Rate limiting
- Error handling

**Key Features:**
- Security headers middleware
- CORS configuration
- API key authentication
- Rate limiting per IP
- Comprehensive input validation

### 2. Feature Engineering

**Hero Features** (`src/features/hero_features.py`)
- Role distribution (Tank/DPS/Support counts)
- Statistical features (pick rates, win rates)
- Hero-level feature extraction

**Team Composition** (`src/features/team_composition.py`)
- Team-level feature extraction
- Role balance metrics
- Embedding-based similarity
- Synergy and counter features

**Hero Embeddings** (`src/features/embeddings.py`)
- Hero2Vec implementation using PyTorch
- Co-occurrence pattern learning
- 32-dimensional embeddings per hero

### 3. Model Layer

**Baseline Model** (`src/models/baseline.py`)
- Logistic regression baseline
- Quick training and evaluation
- Performance floor reference

**XGBoost Model** (`src/models/xgboost_model.py`)
- Gradient boosting classifier
- Feature importance extraction
- Hyperparameter tuning support

**Neural Network** (`src/models/neural_net.py`)
- PyTorch-based MLP
- Uses hero embeddings as input
- Dropout regularization

**Ensemble Model** (`src/models/ensemble.py`)
- Combines XGBoost and Neural Network
- Weighted voting
- Calibrated probability outputs

### 4. Data Pipeline

**Data Collection** (`src/data/collect_matches.py`)
- Synthetic data generation
- API client (placeholder for real data)
- Data validation

**Preprocessing** (`src/data/preprocess.py`)
- Data cleaning
- Feature normalization
- Train/validation/test split

### 5. MLOps

**MLflow Integration**
- Experiment tracking
- Model registry
- Artifact storage

**DVC Pipelines**
- Data versioning
- Reproducible pipelines
- Dependency tracking

## Data Flow

### Training Pipeline

1. **Data Collection**: Generate or collect match data
2. **Preprocessing**: Clean and validate data
3. **Feature Engineering**: Extract features from team compositions
4. **Embedding Training**: Train Hero2Vec embeddings
5. **Model Training**: Train baseline, XGBoost, and Neural Network models
6. **Ensemble Creation**: Combine models into ensemble
7. **Evaluation**: Evaluate on test set
8. **Model Registration**: Register in MLflow

### Inference Pipeline

1. **Request Validation**: Validate team compositions
2. **Feature Extraction**: Extract features from input teams
3. **Model Inference**: Run ensemble model
4. **Response Formatting**: Format predictions
5. **Caching**: Cache results (if enabled)

## Security Architecture

### Authentication
- API key-based authentication
- Keys stored as environment variables
- Secret management with Pydantic SecretStr

### Rate Limiting
- Token bucket algorithm
- Redis-based (with in-memory fallback)
- Configurable limits per IP

### Input Validation
- Pydantic schema validation
- Strict type checking
- Range validation for hero IDs
- Duplicate detection

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security
- Content-Security-Policy

## Deployment Architecture

### Development
- Local FastAPI server with hot reload
- In-memory rate limiting
- SQLite or file-based storage

### Production
- Docker containers
- Gunicorn with multiple workers
- Redis for caching and rate limiting
- Nginx reverse proxy (optional)
- Kubernetes orchestration (optional)

## Scalability Considerations

- **Horizontal Scaling**: Stateless API allows multiple instances
- **Caching**: Redis for prediction caching
- **Model Serving**: Models loaded once per worker process
- **Async Operations**: FastAPI async support for I/O-bound operations
- **Database**: Optional PostgreSQL for storing predictions/history

## Monitoring & Observability

- **Logging**: Structured JSON logging with sensitive data redaction
- **Health Checks**: `/health` endpoint for monitoring
- **Metrics**: Optional Prometheus metrics export
- **Error Tracking**: Comprehensive error logging

