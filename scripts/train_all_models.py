"""Training script to generate data and train all models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from src.data.collect_matches import collect_and_save_matches
from src.data.preprocess import (
    extract_team_compositions,
    load_match_data,
    train_val_test_split,
)
from src.features.embeddings import save_hero_embeddings, train_hero2vec
from src.features.team_composition import TeamCompositionFeatureExtractor
from src.models.baseline import BaselineModel
from src.models.ensemble import EnsembleModel
from src.models.neural_net import NeuralNetModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics

logger = setup_logger(__name__)


def main() -> None:
    """Main training pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_data_path = data_dir / "raw" / "synthetic_matches.csv"
    embeddings_path = data_dir / "embeddings" / "hero_embeddings.npy"
    models_dir = project_root / "models"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    (data_dir / "embeddings").mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    logger.info("Starting training pipeline...")

    # Step 1: Generate synthetic data
    logger.info("Step 1: Generating synthetic match data...")
    if not raw_data_path.exists():
        collect_and_save_matches(raw_data_path, n_matches=10000, random_seed=42)
    else:
        logger.info(f"Data file already exists: {raw_data_path}")

    # Step 2: Load and preprocess data
    logger.info("Step 2: Loading and preprocessing data...")
    df = load_match_data(raw_data_path)
    team1_list, team2_list, y = extract_team_compositions(df)

    # Step 3: Train embeddings
    logger.info("Step 3: Training Hero2Vec embeddings...")
    if not embeddings_path.exists():
        matches = [(t1, t2) for t1, t2 in zip(team1_list, team2_list)]
        embedding_model = train_hero2vec(
            matches, embedding_dim=32, epochs=50, batch_size=64
        )
        embeddings = embedding_model.get_embeddings()
        save_hero_embeddings(embeddings, embeddings_path)
        logger.info(f"Embeddings saved to {embeddings_path}")
    else:
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings from {embeddings_path}")

    # Step 4: Extract features
    logger.info("Step 4: Extracting features...")
    extractor = TeamCompositionFeatureExtractor(embeddings=embeddings)
    X = np.array(
        [
            extractor.extract_feature_vector(t1, t2)
            for t1, t2 in zip(team1_list, team2_list)
        ]
    )
    logger.info(f"Extracted {len(X)} feature vectors of dimension {X.shape[1]}")

    # Step 5: Split data
    logger.info("Step 5: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    # Step 6: Train baseline model
    logger.info("Step 6: Training baseline model...")
    baseline = BaselineModel(random_state=42, max_iter=1000)
    train_acc, train_metrics = baseline.train(X_train, y_train)
    logger.info(f"Baseline training accuracy: {train_acc:.4f}")

    test_pred = baseline.predict(X_test)
    test_proba = baseline.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_pred, test_proba)
    logger.info(f"Baseline test accuracy: {test_metrics['accuracy']:.4f}")

    baseline.save(models_dir / "baseline.pkl")

    # Step 7: Train XGBoost model
    logger.info("Step 7: Training XGBoost model...")
    xgboost = XGBoostModel(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        early_stopping_rounds=10,
    )
    train_acc, train_metrics = xgboost.train(X_train, y_train, X_val, y_val)
    logger.info(f"XGBoost training accuracy: {train_acc:.4f}")

    test_pred = xgboost.predict(X_test)
    test_proba = xgboost.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_pred, test_proba)
    logger.info(f"XGBoost test accuracy: {test_metrics['accuracy']:.4f}")

    xgboost.save(models_dir / "xgboost_model.pkl")

    # Step 8: Train Neural Network
    logger.info("Step 8: Training Neural Network model...")
    neural_net = NeuralNetModel(
        input_dim=X.shape[1],
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        random_state=42,
        device="cpu",
    )
    train_acc, train_metrics = neural_net.train(X_train, y_train, X_val, y_val)
    logger.info(f"Neural Network training accuracy: {train_acc:.4f}")

    test_pred = neural_net.predict(X_test)
    test_proba = neural_net.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_pred, test_proba)
    logger.info(f"Neural Network test accuracy: {test_metrics['accuracy']:.4f}")

    neural_net.save(models_dir / "neural_net_model.pt")

    # Step 9: Create and save ensemble
    logger.info("Step 9: Creating ensemble model...")
    ensemble = EnsembleModel(
        xgboost_model=xgboost,
        neural_net_model=neural_net,
        xgboost_weight=0.6,
        neural_net_weight=0.4,
    )

    test_pred = ensemble.predict(X_test)
    test_proba = ensemble.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_pred, test_proba)
    logger.info(f"Ensemble test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Ensemble test F1: {test_metrics['f1']:.4f}")

    ensemble.save(models_dir / "ensemble")

    logger.info("Training pipeline completed successfully!")
    logger.info(f"Models saved to {models_dir}")


if __name__ == "__main__":
    main()

