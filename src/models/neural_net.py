"""Neural network model for match prediction."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MatchPredictorNN(nn.Module):
    """Neural network for match prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize neural network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output probabilities (batch_size, 1)
        """
        return self.network(x)


class NeuralNetModel:
    """Neural network model wrapper."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        """
        Initialize neural network model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            random_state: Random seed
            device: Device to train on ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = MatchPredictorNN(input_dim, hidden_dims, dropout).to(self.device)
        self.is_trained = False

    def train(
        self,
        X_train: NDArray[np.float_],
        y_train: NDArray[np.int_],
        X_val: NDArray[np.float_] | None = None,
        y_val: NDArray[np.int_] | None = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the neural network.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Tuple of (training_accuracy, metrics_dict)
        """
        logger.info(f"Training neural network on {len(X_train)} samples")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on training data
        self.model.eval()
        with torch.no_grad():
            y_pred_proba = self.model(X_train_tensor).cpu().numpy().flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)

        train_accuracy = float(np.mean(y_pred == y_train))

        from src.utils.metrics import calculate_metrics

        metrics = calculate_metrics(y_train, y_pred, y_pred_proba)
        self.is_trained = True

        logger.info(f"Neural network training complete. Accuracy: {train_accuracy:.4f}")

        return train_accuracy, metrics

    def predict(self, X: NDArray[np.float_]) -> NDArray[np.int_]:
        """
        Predict match outcomes.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            y_proba = self.model(X_tensor).cpu().numpy().flatten()
            y_pred = (y_proba >= 0.5).astype(int)

        return y_pred

    def predict_proba(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Predict match outcome probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, 2) for classes [0, 1]

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            proba_team1 = self.model(X_tensor).cpu().numpy().flatten()

        # Return in format [prob_class_0, prob_class_1]
        proba = np.column_stack([1 - proba_team1, proba_team1])
        return proba

    def save(self, model_path: Path) -> None:
        """
        Save model to disk.

        Args:
            model_path: Path to save model file
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
            model_path,
        )
        logger.info(f"Neural network model saved to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = "cpu") -> "NeuralNetModel":
        """
        Load model from disk.

        Args:
            model_path: Path to model file
            device: Device to load model on

        Returns:
            Loaded NeuralNetModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        instance = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.is_trained = True
        logger.info(f"Neural network model loaded from {model_path}")

        return instance
