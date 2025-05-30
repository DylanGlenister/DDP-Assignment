"""
	File for training machine learning models on acrhery scores.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

import shared

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ArcheryDataset(Dataset):
	"""Custom dataset for archery score prediction"""

	def __init__(
		self,
		_features: np.ndarray,
		_targets: np.ndarray,
		_archer_encoded: np.ndarray
	):
		self.features = torch.FloatTensor(_features)
		self.targets = torch.FloatTensor(_targets)
		self.archer_encoded = torch.LongTensor(_archer_encoded)

	def __len__(self) -> int:
		return len(self.features)

	def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
		return {
			'features': self.features[_idx],
			'targets': self.targets[_idx],
			'archer': self.archer_encoded[_idx]
		}


class ArcheryRNN(nn.Module):
	"""Unified RNN model supporting LSTM, GRU, and Transformer architectures"""

	def __init__(
		self,
		_sequence_length: int,
		_n_archers: int,
		_model_type: str = shared.MODEL_TYPE_LSTM,
		_hidden_dim: int = 64,
		_n_layers: int = 2,
		_dropout: float = 0.2,
		_n_heads: int = 8
	):
		super().__init__()

		self.hidden_dim = _hidden_dim
		self.n_layers = _n_layers
		self.sequence_length = _sequence_length
		self.model_type = _model_type.lower()

		# Embedding layer for archer ID
		self.archer_embedding = nn.Embedding(_n_archers, 32)

		# RNN input size: score + archer embedding
		rnn_input_size = 1 + 32

		# Create the appropriate model type
		if self.model_type == shared.MODEL_TYPE_LSTM:
			self.rnn = nn.LSTM(rnn_input_size, _hidden_dim, _n_layers,
							  batch_first=True, dropout=_dropout if _n_layers > 1 else 0)
		elif self.model_type == shared.MODEL_TYPE_GRU:
			self.rnn = nn.GRU(rnn_input_size, _hidden_dim, _n_layers,
							 batch_first=True, dropout=_dropout if _n_layers > 1 else 0)
		elif self.model_type == shared.MODEL_TYPE_TFMR:
			d_model = self._adjust_d_model_for_heads(_hidden_dim, _n_heads)
			self.input_projection = nn.Linear(rnn_input_size, d_model)
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=d_model,
				nhead=_n_heads,
				dim_feedforward=d_model * 2,
				dropout=_dropout,
				batch_first=True
			)
			self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=_n_layers)
			self.hidden_dim = d_model

		# Output layers
		self.dropout = nn.Dropout(_dropout)
		self.fc = nn.Linear(self.hidden_dim, 1)

	def _adjust_d_model_for_heads(self, _hidden_dim: int, _n_heads: int) -> int:
		"""Adjust d_model to be divisible by n_heads"""
		if _hidden_dim % _n_heads != 0:
			d_model = ((_hidden_dim // _n_heads) + 1) * _n_heads
			print(f'Adjusted d_model to {d_model} to be divisible by {_n_heads} heads')
			return d_model
		return _hidden_dim

	def forward(self, _x: torch.Tensor, _archer: torch.Tensor) -> torch.Tensor:
		# Get archer embedding and expand to sequence length
		archer_emb = self.archer_embedding(_archer)
		archer_emb = archer_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)

		# Concatenate score sequence with archer embedding
		rnn_input = torch.cat([_x, archer_emb], dim=2)

		# Forward pass through the selected architecture
		if self.model_type in [shared.MODEL_TYPE_LSTM, shared.MODEL_TYPE_GRU]:
			rnn_out, _ = self.rnn(rnn_input)
			output = self.dropout(rnn_out[:, -1, :])
		#elif self.model_type == shared.MODEL_TYPE_TFMR:
		else:
			projected_input = self.input_projection(rnn_input)
			transformer_out = self.rnn(projected_input)
			output = self.dropout(transformer_out[:, -1, :])

		return self.fc(output)


class DataProcessor:
	"""Handles data loading and preprocessing"""

	def __init__(self, _sequence_length: int = 12):
		self.sequence_length = _sequence_length
		self.sequence_scaler = StandardScaler()
		self.target_scaler = StandardScaler()
		self.archer_encoder = LabelEncoder()

	def load_data(self, _filepath: str) -> pd.DataFrame:
		"""Load and preprocess the archery data"""
		print('Loading data...')
		df = pd.read_csv(_filepath)
		df[shared.COLUMN_DATE] = pd.to_datetime(df[shared.COLUMN_DATE])
		df = df.sort_values([shared.COLUMN_ARCHER_ID, shared.COLUMN_DATE])

		print(f'Data shape: {df.shape}')
		print(f'Unique archers: {df[shared.COLUMN_ARCHER_ID].nunique()}')
		print(f'Date range: {df[shared.COLUMN_DATE].min()} to {df[shared.COLUMN_DATE].max()}')
		print(f'Score range: {df[shared.COLUMN_SCORE].min():.4f} to {df[shared.COLUMN_SCORE].max():.4f}')

		return df

	def create_sequences(self, _df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
		"""Create sequences for training"""
		print('Creating sequences...')
		sequences: list[np.ndarray] = []
		targets: list[float] = []
		archer_list: list = []

		for archer_id, group in _df.groupby(shared.COLUMN_ARCHER_ID):
			scores = group[shared.COLUMN_SCORE].values.astype(float)

			if len(scores) <= self.sequence_length:
				print(f'Skipping archer {archer_id}: only {len(scores)} scores (need >{self.sequence_length})')
				continue

			for i in range(len(scores) - self.sequence_length):
				sequences.append(scores[i:i + self.sequence_length])
				targets.append(scores[i + self.sequence_length])
				archer_list.append(archer_id)

		print(f'Created {len(sequences)} sequences from {len(set(archer_list))} archers')
		return np.array(sequences), np.array(targets), archer_list

	def prepare_data(self, _filepath: str, _test_size: float = 0.2) -> tuple[ArcheryDataset, ArcheryDataset]:
		"""Prepare data for training"""
		df = self.load_data(_filepath)
		sequences, targets, archer_list = self.create_sequences(df)

		# Encode and scale data
		archer_encoded = self.archer_encoder.fit_transform(archer_list)
		sequences_scaled = self.sequence_scaler.fit_transform(sequences)
		targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

		# Split data
		X_train, X_test, y_train, y_test, archer_train, archer_test = train_test_split(
			sequences_scaled, targets_scaled, archer_encoded,
			test_size=_test_size, random_state=42, stratify=archer_encoded
		)

		# Reshape for RNN input
		X_train = X_train.reshape(-1, self.sequence_length, 1)
		X_test = X_test.reshape(-1, self.sequence_length, 1)

		# Create datasets
		train_dataset = ArcheryDataset(X_train, y_train, archer_train)
		test_dataset = ArcheryDataset(X_test, y_test, archer_test)

		print(f'Training samples: {len(train_dataset)}')
		print(f'Test samples: {len(test_dataset)}')
		print(f'Number of archers: {len(self.archer_encoder.classes_)}')

		return train_dataset, test_dataset


class ModelTrainer:
	"""Handles model training and evaluation"""

	def __init__(self, _model: nn.Module, _device: torch.device):
		self.model = _model
		self.device = _device
		self.optimizer = optim.Adam(_model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

	def train(
		self,
		_train_dataset: ArcheryDataset,
		_test_dataset: ArcheryDataset,
		_epochs: int = 100,
		_batch_size: int = 64
	) -> tuple[list[float], list[float]]:
		"""Train the model"""
		train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)
		test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

		train_losses: list[float] = []
		test_losses: list[float] = []

		print(f'Training model...')

		for epoch in range(_epochs):
			# Training phase
			self.model.train()
			train_loss = self._run_epoch(train_loader, _training=True)

			# Validation phase
			self.model.eval()
			with torch.no_grad():
				test_loss = self._run_epoch(test_loader, _training=False)

			train_losses.append(train_loss)
			test_losses.append(test_loss)

			if (epoch + 1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

		return train_losses, test_losses

	def _run_epoch(self, _data_loader: DataLoader, _training: bool = True) -> float:
		"""Run a single epoch"""
		total_loss: float = 0

		for batch in _data_loader:
			features = batch['features'].to(self.device)
			targets = batch['targets'].to(self.device)
			archer = batch['archer'].to(self.device)

			if _training:
				self.optimizer.zero_grad()

			outputs = self.model(features, archer)
			loss = self.criterion(outputs.squeeze(), targets)

			if _training:
				loss.backward()
				self.optimizer.step()

			total_loss += loss.item()

		return total_loss / len(_data_loader)


class ArcheryPredictor:
	"""Main class for archery score prediction"""

	def __init__(self, _sequence_length: int = 12, _model_type: str = shared.MODEL_TYPE_LSTM):
		self.sequence_length = _sequence_length
		self.model_type = _model_type.lower()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.data_processor = DataProcessor(_sequence_length)
		self.model: ArcheryRNN | None = None
		self.trainer: ModelTrainer | None = None

		print(f'Using device: {self.device}')

	def load_or_train(
		self,
		_data_filepath: str | None = None,
		_force_retrain: bool = False,
		**_train_params
	) -> bool:
		"""Load existing model or train a new one if loading fails"""
		if _data_filepath is None:
			_data_filepath = shared.PATH_DATASET

		print(f'Processing {shared.MODEL_DISPLAY_NAMES[self.model_type]} Model')

		# Try to load existing model unless forced to retrain
		if not _force_retrain and self._load_model():
			print(f'✓ Successfully loaded existing {shared.MODEL_DISPLAY_NAMES[self.model_type]} model')
			print(f'Model ready for predictions. Archer count: {len(self.data_processor.archer_encoder.classes_)}')
			return True

		# Train new model
		print(f'Training new {shared.MODEL_DISPLAY_NAMES[self.model_type]} model...')
		return self._train_new_model(_data_filepath, **_train_params)

	def _train_new_model(
		self,
		_data_filepath: str,
		_hidden_dim: int = 64,
		_n_layers: int = 2,
		_dropout: float = 0.2,
		_n_heads: int = 8,
		_epochs: int = 50,
		_batch_size: int = 64,
		_test_size: float = 0.2,
		_show_plot: bool = False
	) -> bool:
		"""Train a new model"""
		try:
			# Prepare data
			train_dataset, test_dataset = self.data_processor.prepare_data(_data_filepath, _test_size)

			# Create model
			n_archers = len(self.data_processor.archer_encoder.classes_)
			self.model = ArcheryRNN(
				self.sequence_length, n_archers, self.model_type,
				_hidden_dim, _n_layers, _dropout, _n_heads
			).to(self.device)

			print(f'Created {shared.MODEL_DISPLAY_NAMES[self.model_type]} model with {sum(p.numel() for p in self.model.parameters())} parameters')

			# Train model
			self.trainer = ModelTrainer(self.model, self.device)
			train_losses, test_losses = self.trainer.train(
				train_dataset, test_dataset, _epochs, _batch_size
			)

			# Save and plot results
			self._save_model()
			if _show_plot:
				self._plot_training_history(train_losses, test_losses)

			print(f'✓ {shared.MODEL_DISPLAY_NAMES[self.model_type]} model trained and saved successfully')
			return True

		except Exception as e:
			print(f'✗ Error training {shared.MODEL_DISPLAY_NAMES[self.model_type]} model: {e}')
			return False

	def predict_score(self, _archer_id: int | str, _recent_scores: list[float] | None = None) -> float:
		"""Predict next score for a given archer"""
		assert(self.model is not None)
		self.model.eval()

		# Encode archer ID
		archer_encoded = self.data_processor.archer_encoder.transform([_archer_id])[0]  # type: ignore

		# Prepare recent scores
		if _recent_scores is None:
			print('Warning: No recent score data provided. Using dummy sequence.')
			_recent_scores = [0.7] * self.sequence_length

		recent_scores = self._prepare_score_sequence(_recent_scores)

		# Scale and predict
		sequence_scaled = self.data_processor.sequence_scaler.transform([recent_scores])
		features = torch.FloatTensor(sequence_scaled).reshape(1, self.sequence_length, 1).to(self.device)
		archer_tensor = torch.LongTensor([archer_encoded]).to(self.device)

		with torch.no_grad():
			prediction_scaled = self.model(features, archer_tensor)

		# Inverse transform and clamp
		prediction = self.data_processor.target_scaler.inverse_transform([[prediction_scaled.cpu().item()]])[0][0]
		return max(0.0, min(1.0, prediction))

	def _prepare_score_sequence(self, _recent_scores: list[float]) -> list[float]:
		"""Prepare score sequence for prediction"""
		if len(_recent_scores) < self.sequence_length:
			mean_score = float(np.mean(_recent_scores) if _recent_scores else 0.7)
			padding = [mean_score] * (self.sequence_length - len(_recent_scores))
			return padding + list(_recent_scores)
		elif len(_recent_scores) > self.sequence_length:
			return _recent_scores[-self.sequence_length:]
		return _recent_scores

	def _save_model(self):
		"""Save the trained model"""
		filepath = f'{shared.PATH_MODELS}archery_{self.model_type}_model.pt'
		assert(self.model is not None)
		assert(self.trainer is not None)
		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.trainer.optimizer.state_dict(),
			'data_processor': self.data_processor,
			'sequence_length': self.sequence_length,
			'model_type': self.model_type,
			'model_params': {
				'hidden_dim': self.model.hidden_dim,
				'n_layers': self.model.n_layers,
				'n_heads': getattr(self.model, 'n_heads', 8)
			}
		}
		torch.save(checkpoint, filepath)
		print(f'Model saved to {filepath}')

	def _load_model(self) -> bool:
		"""Load a trained model"""
		filepath = f'{shared.PATH_MODELS}archery_{self.model_type}_model.pt'
		try:
			checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

			# Restore data processor and model parameters
			self.data_processor = checkpoint['data_processor']
			model_params = checkpoint['model_params']

			# Recreate model
			n_archers = len(self.data_processor.archer_encoder.classes_)
			self.model = ArcheryRNN(
				self.sequence_length, n_archers, self.model_type,
				model_params['hidden_dim'], model_params['n_layers'], 0.2,
				model_params.get('n_heads', 8)
			).to(self.device)

			self.model.load_state_dict(checkpoint['model_state_dict'])
			return True

		except FileNotFoundError:
			print(f'Model file {filepath} not found.')
			return False
		except Exception as e:
			print(f'Error loading model: {e}')
			return False

	def _plot_training_history(self, _train_losses: list[float], _test_losses: list[float]):
		"""Plot training history"""
		plt.figure(figsize=(10, 6))
		plt.plot(_train_losses, label='Training Loss')
		plt.plot(_test_losses, label='Validation Loss')
		plt.title(f'{shared.MODEL_DISPLAY_NAMES[self.model_type]} Model Training History')
		plt.xlabel('Epoch')
		plt.ylabel('Loss (MSE)')
		plt.legend()
		plt.grid(True)
		plt.show()

	def get_available_archers(self) -> list:
		"""Get list of available archer IDs"""
		return list(self.data_processor.archer_encoder.classes_)


def main():
	"""Main function demonstrating usage of all model types"""
	models: dict[str, ArcheryPredictor] = {}

	for model_type in shared.MODEL_TYPES:
		predictor = ArcheryPredictor(_sequence_length=12, _model_type=model_type)

		if predictor.load_or_train(_epochs=50, _batch_size=64):
			models[model_type] = predictor
		else:
			print(f'Failed to initialize {model_type} model')

	# Demonstration if all models loaded successfully
	for _, model in models.items():
		print('\n' + '=' * 60)
		print('All Models Ready for Predictions')
		print('=' * 60)

		# Sample prediction demonstration
		sample_scores = [
			0.7615573632329024, 0.6122834217765443, 0.7433769949815001,
			0.7108834281844016, 0.7174447752290504, 0.6029931475517498,
			0.8136260904112261, 0.742366237718169, 0.6288655285771527,
			0.7277966117076704, 0.5661138417053742, 0.6897291459339614
		]

		prediction = model.predict_score(0, sample_scores)
		print(f'Sample prediction for Archer 0: {prediction:.4f}')
		print(f'Available archers: {len(model.get_available_archers())}')


if __name__ == '__main__':
	main()
