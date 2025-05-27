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

warnings.filterwarnings('ignore')

PATH_DATASET = './data.csv'

# Column names for the archery dataset
COLUMN_ARCHER_ID = 'ArcherID'
COLUMN_DATE = 'Date'
COLUMN_SCORE = 'Score'

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ArcheryDataset(Dataset):
	'''Custom dataset for archery score prediction'''

	def __init__(self, _features, _targets, _archer_encoded):
		self.features = torch.FloatTensor(_features)
		self.targets = torch.FloatTensor(_targets)
		self.archer_encoded = torch.LongTensor(_archer_encoded)

	def __len__(self):
		return len(self.features)

	def __getitem__(self, _idx):
		return {
			'features': self.features[_idx],
			'targets': self.targets[_idx],
			'archer': self.archer_encoded[_idx]
		}

class ArcheryRNN(nn.Module):
	'''Unified RNN model supporting LSTM, GRU, and Transformer architectures for archery score prediction'''

	def __init__(self, _sequence_length, _n_archers, _model_type='lstm',
				 _hidden_dim=64, _n_layers=2, _dropout=0.2, _n_heads=8):
		super(ArcheryRNN, self).__init__()

		self.hidden_dim = _hidden_dim
		self.n_layers = _n_layers
		self.sequence_length = _sequence_length
		self.model_type = _model_type.lower()

		# Embedding layer for archer ID
		self.archer_embedding = nn.Embedding(_n_archers, 32)

		# RNN input size: score + archer embedding
		rnn_input_size = 1 + 32  # score + archer_emb = 33

		# Create the appropriate model type
		if self.model_type == 'lstm':
			self.rnn = nn.LSTM(rnn_input_size, _hidden_dim, _n_layers,
							  batch_first=True, dropout=_dropout if _n_layers > 1 else 0)
		elif self.model_type == 'gru':
			self.rnn = nn.GRU(rnn_input_size, _hidden_dim, _n_layers,
							 batch_first=True, dropout=_dropout if _n_layers > 1 else 0)
		elif self.model_type == 'transformer':
			# For transformer, we need d_model to be divisible by n_heads
			d_model = _hidden_dim

			# Ensure d_model is divisible by n_heads
			if d_model % _n_heads != 0:
				# Adjust d_model to be divisible by n_heads
				d_model = ((d_model // _n_heads) + 1) * _n_heads
				print(f'Adjusted d_model to {d_model} to be divisible by {_n_heads} heads')

			# Project input to d_model dimensions
			self.input_projection = nn.Linear(rnn_input_size, d_model)

			# Transformer encoder layer
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=d_model,
				nhead=_n_heads,
				dim_feedforward=d_model * 2,
				dropout=_dropout,
				batch_first=True
			)
			self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=_n_layers)
			# Update hidden_dim to match d_model for consistency
			self.hidden_dim = d_model
		else:
			raise ValueError(f'model_type must be "lstm", "gru", or "transformer"')

		# Output layers
		self.dropout = nn.Dropout(_dropout)
		self.fc = nn.Linear(self.hidden_dim, 1)

	def forward(self, _x, _archer):
		# Get archer embedding
		archer_emb = self.archer_embedding(_archer)  # (batch_size, 32)

		# Expand archer embedding to match sequence length
		archer_emb = archer_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)

		# Concatenate score sequence with archer embedding
		rnn_input = torch.cat([_x, archer_emb], dim=2)

		output = None

		# Forward pass through the selected architecture
		if self.model_type in ['lstm', 'gru']:
			rnn_out, _ = self.rnn(rnn_input)
			# Use the last output for prediction
			output = self.dropout(rnn_out[:, -1, :])
		elif self.model_type == 'transformer':
			# Project input to d_model dimensions
			projected_input = self.input_projection(rnn_input)
			# Transformer doesn't need hidden states
			transformer_out = self.rnn(projected_input)
			# Use the last output
			output = self.dropout(transformer_out[:, -1, :])

		if output is None:
			raise TypeError('Error failure trying to set forward output')

		output = self.fc(output)
		return output

class ArcheryPredictor:
	'''Main class for archery score prediction'''

	def __init__(self, _sequence_length=12, _model_type='lstm'):
		self.sequence_length = _sequence_length
		self.model_type = _model_type.lower()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Preprocessing objects
		self.sequence_scaler = StandardScaler()
		self.target_scaler = StandardScaler()
		self.archer_encoder = LabelEncoder()

		# Model and training objects
		self.model = None
		self.optimizer = None
		self.criterion = nn.MSELoss()

		# Model dimensions (will be set during data preparation or loading)
		self.n_archers = None

		print(f'Using device: {self.device}')

	def load_and_preprocess_data(self, _filepath=PATH_DATASET):
		'''Load and preprocess the archery data'''
		print('Loading data...')
		df = pd.read_csv(_filepath)

		# Convert date to datetime for proper sorting
		df[COLUMN_DATE] = pd.to_datetime(df[COLUMN_DATE])

		# Sort by archer and date to ensure chronological order
		df = df.sort_values([COLUMN_ARCHER_ID, COLUMN_DATE])

		print(f'Data shape: {df.shape}')
		print(f'Unique archers: {df[COLUMN_ARCHER_ID].nunique()}')
		print(f'Date range: {df[COLUMN_DATE].min()} to {df[COLUMN_DATE].max()}')

		return df

	def create_sequences(self, _df):
		'''Create sequences for training'''
		print('Creating sequences...')

		sequences = []
		targets = []
		archer_list = []

		# Group by archer to create sequences
		for archer_id, group in _df.groupby(COLUMN_ARCHER_ID):
			# Get scores in chronological order
			scores = group[COLUMN_SCORE].values.astype(float)

			# Skip archers with insufficient data
			if len(scores) <= self.sequence_length:
				continue

			# Create sequences for this archer
			for i in range(len(scores) - self.sequence_length):
				seq = scores[i:i + self.sequence_length]
				target = scores[i + self.sequence_length]

				sequences.append(seq)
				targets.append(target)
				archer_list.append(archer_id)

		print(f'Created {len(sequences)} sequences from {len(set(archer_list))} archers')
		return np.array(sequences), np.array(targets), archer_list

	def prepare_data(self, _filepath=PATH_DATASET, _test_size=0.2):
		'''Prepare data for training'''
		# Load data
		df = self.load_and_preprocess_data(_filepath)

		# Create sequences
		sequences, targets, archer_list = self.create_sequences(df)

		if len(sequences) == 0:
			raise ValueError('No sequences created. Check that archers have sufficient historical data.')

		# Encode archer IDs
		archer_encoded = self.archer_encoder.fit_transform(archer_list)

		# Scale the sequences and targets separately
		sequences_scaled = self.sequence_scaler.fit_transform(sequences)
		targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

		# Split data, stratifying by archer to ensure balanced representation
		X_train, X_test, y_train, y_test, archer_train, archer_test = train_test_split(
			sequences_scaled, targets_scaled, archer_encoded,
			test_size=_test_size, random_state=42, stratify=archer_encoded
		)

		# Reshape for RNN input (batch_size, sequence_length, features)
		X_train = X_train.reshape(-1, self.sequence_length, 1)
		X_test = X_test.reshape(-1, self.sequence_length, 1)

		# Create datasets
		train_dataset = ArcheryDataset(X_train, y_train, archer_train)
		test_dataset = ArcheryDataset(X_test, y_test, archer_test)

		self.n_archers = len(self.archer_encoder.classes_)

		print(f'Training samples: {len(train_dataset)}')
		print(f'Test samples: {len(test_dataset)}')
		print(f'Number of archers: {self.n_archers}')

		return train_dataset, test_dataset

	def create_model(self, _hidden_dim=64, _n_layers=2, _dropout=0.2, _n_heads=8):
		'''Create LSTM, GRU, or Transformer model'''
		self.model = ArcheryRNN(
			self.sequence_length, self.n_archers,
			self.model_type, _hidden_dim, _n_layers, _dropout, _n_heads
		)

		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		print(f'Created {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters')

	def train(self, _train_dataset, _test_dataset, _epochs=100, _batch_size=64):
		'''Train the model'''
		train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)
		test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

		train_losses: list[float] = []
		test_losses: list[float] = []

		if self.model is None:
			raise ValueError('Cannot train, model has not been set')

		if self.optimizer is None:
			raise ValueError('Cannot train, optimizer has not been set')

		print(f'Training {self.model_type.upper()} model...')

		for epoch in range(_epochs):
			# Training
			self.model.train()
			train_loss: float = 0

			for batch in train_loader:
				features = batch['features'].to(self.device)
				targets = batch['targets'].to(self.device)
				archer = batch['archer'].to(self.device)

				self.optimizer.zero_grad()
				outputs = self.model(features, archer)
				loss = self.criterion(outputs.squeeze(), targets)
				loss.backward()
				self.optimizer.step()

				train_loss += loss.item()

			# Validation
			self.model.eval()
			test_loss: float = 0

			with torch.no_grad():
				for batch in test_loader:
					features = batch['features'].to(self.device)
					targets = batch['targets'].to(self.device)
					archer = batch['archer'].to(self.device)

					outputs = self.model(features, archer)
					loss = self.criterion(outputs.squeeze(), targets)
					test_loss += loss.item()

			train_loss /= len(train_loader)
			test_loss /= len(test_loader)

			train_losses.append(train_loss)
			test_losses.append(test_loss)

			if (epoch + 1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

		return train_losses, test_losses

	def predict_score(self, _archer_id, _recent_scores=None):
		'''
		Predict next score for a given archer

		Args:
			archer_id: Archer identifier
			recent_scores: List of recent scores for sequence (optional)

		Returns:
			Predicted score
		'''
		if self.model is None:
			raise ValueError('Cannot predict, model has not been set')

		self.model.eval()

		# Encode archer ID
		try:
			archer_encoded = self.archer_encoder.transform([_archer_id])[0]
		except ValueError as e:
			raise ValueError(f'Unknown archer ID: {e}')

		# If recent_scores not provided, use dummy sequence (this is not ideal for real prediction)
		if _recent_scores is None:
			print('Warning: No recent score data provided. Using dummy sequence.')
			_recent_scores = [0.0] * self.sequence_length

		if len(_recent_scores) < self.sequence_length:
			# Pad with zeros if not enough data
			_recent_scores = [0.0] * (self.sequence_length - len(_recent_scores)) + list(_recent_scores)
		elif len(_recent_scores) > self.sequence_length:
			# Take the last sequence_length values
			_recent_scores = _recent_scores[-self.sequence_length:]

		# Scale the input sequence using sequence_scaler
		sequence_scaled = self.sequence_scaler.transform([_recent_scores])

		# Prepare input tensors
		features = torch.FloatTensor(sequence_scaled).reshape(1, self.sequence_length, 1).to(self.device)
		archer_tensor = torch.LongTensor([archer_encoded]).to(self.device)

		# Make prediction
		with torch.no_grad():
			prediction_scaled = self.model(features, archer_tensor)

		# Inverse transform using target_scaler to get actual score
		prediction = self.target_scaler.inverse_transform([[prediction_scaled.cpu().item()]])[0][0]

		return max(0, prediction)  # Ensure non-negative score

	def save_model(self, _filepath=None):
		'''Save the trained model and preprocessors'''
		if _filepath is None:
			_filepath = f'archery_{self.model_type}_model.pt'

		if self.model is None:
			raise ValueError('No model to save. Train a model first.')

		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
			'sequence_scaler': self.sequence_scaler,
			'target_scaler': self.target_scaler,
			'archer_encoder': self.archer_encoder,
			'sequence_length': self.sequence_length,
			'model_type': self.model_type,
			'n_archers': self.n_archers,
			'model_params': {
				'hidden_dim': self.model.hidden_dim,
				'n_layers': self.model.n_layers,
				'n_heads': getattr(self.model, 'n_heads', 8)  # Default for non-transformer models
			}
		}

		torch.save(checkpoint, _filepath)
		print(f'Model saved to {_filepath}')

	def load_model(self, _filepath=None):
		'''Load a trained model and preprocessors'''
		if _filepath is None:
			_filepath = f'archery_{self.model_type}_model.pt'

		try:
			checkpoint = torch.load(_filepath, map_location=self.device, weights_only=False)

			# Load preprocessors
			self.sequence_scaler = checkpoint['sequence_scaler']
			self.target_scaler = checkpoint['target_scaler']
			self.archer_encoder = checkpoint['archer_encoder']
			self.sequence_length = checkpoint['sequence_length']
			self.n_archers = checkpoint['n_archers']

			# Recreate model with saved parameters
			model_params = checkpoint['model_params']
			self.create_model(
				_hidden_dim=model_params['hidden_dim'],
				_n_layers=model_params['n_layers'],
				_n_heads=model_params.get('n_heads', 8)  # Default for backward compatibility
			)
			assert(self.model is not None)

			# Load model state
			self.model.load_state_dict(checkpoint['model_state_dict'])

			# Load optimizer state if available
			if checkpoint['optimizer_state_dict'] is not None and self.optimizer is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

			print(f'Model loaded from {_filepath}')
			return True

		except FileNotFoundError:
			print(f'Model file {_filepath} not found.')
			return False
		except Exception as e:
			print(f'Error loading model: {e}')
			return False

	def plot_training_history(self, _train_losses, _test_losses):
		'''Plot training history'''
		plt.figure(figsize=(10, 6))
		plt.plot(_train_losses, label='Training Loss')
		plt.plot(_test_losses, label='Validation Loss')
		plt.title(f'{self.model_type.upper()} Model Training History')
		plt.xlabel('Epoch')
		plt.ylabel('Loss (MSE)')
		plt.legend()
		plt.grid(True)
		plt.show()

# Example usage
def main():
	'''
	Main function that attempts to load existing models, or trains new ones if loading fails
	'''

	def load_or_train_model(model_type):
		'''Helper function to load existing model or train a new one'''
		print('=' * 60)
		print(f'Processing {model_type.upper()} Model')
		print('=' * 60)

		predictor = ArcheryPredictor(_sequence_length=12, _model_type=model_type)

		# Try to load existing model
		model_loaded = predictor.load_model()

		if model_loaded:
			print(f'✓ Successfully loaded existing {model_type.upper()} model')

			# Verify model is working by checking if we can make a dummy prediction
			try:
				if hasattr(predictor, 'n_archers') and predictor.n_archers is not None:
					print(f'Model ready for predictions. Archer count: {predictor.n_archers}')
				else:
					print('Warning: Model loaded but some parameters may be missing')
			except Exception as e:
				print(f'Warning: Loaded model may have issues: {e}')

		else:
			print(f'✗ No existing {model_type.upper()} model found. Training new model...')

			try:
				# Prepare data
				train_dataset, test_dataset = predictor.prepare_data(PATH_DATASET)

				# Create and train model
				predictor.create_model(_hidden_dim=64, _n_layers=2, _dropout=0.2)
				print(f'Training {model_type.upper()} model...')
				assert(predictor is not None)
				train_losses, test_losses = predictor.train(
					train_dataset, test_dataset,
					_epochs=50, _batch_size=64
				)

				# Save the trained model
				predictor.save_model()

				# Plot training history
				predictor.plot_training_history(train_losses, test_losses)

				print(f'✓ {model_type.upper()} model trained and saved successfully')

			except FileNotFoundError:
				print(f'✗ Error: Data file not found at {PATH_DATASET}. Please ensure the data file exists.')
				return None
			except Exception as e:
				print(f'✗ Error training {model_type.upper()} model: {e}')
				return None

		return predictor

	# Load or train LSTM model
	lstm_predictor = load_or_train_model('lstm')

	# Load or train GRU model
	gru_predictor = load_or_train_model('gru')

	# Load or train Transformer model
	transformer_predictor = load_or_train_model('transformer')

	# Example predictions (only if all models loaded successfully)
	if all(model is not None for model in [lstm_predictor, gru_predictor, transformer_predictor]):
		print('\n' + '=' * 60)
		print('All Models Ready for Predictions')
		print('=' * 60)

		print('All three models (LSTM, GRU, Transformer) are ready for use!')
		print('\nTo make predictions, use:')
		print('predictor.predict_score(archer_id, recent_scores)')
		print('\nExample:')
		print('recent_scores = [85, 87, 82, 89, 91, 88, 86, 90, 93, 87, 89, 92]')
		print('prediction = predictor.predict_score("archer_123", recent_scores)')

		# Show available archers if models are loaded
		try:
			if hasattr(lstm_predictor, 'archer_encoder') and hasattr(lstm_predictor.archer_encoder, 'classes_'):
				print(f'\nAvailable Archers ({len(lstm_predictor.archer_encoder.classes_)}): {list(lstm_predictor.archer_encoder.classes_[:10])}{"..." if len(lstm_predictor.archer_encoder.classes_) > 10 else ""}')
		except:
			pass

	return lstm_predictor, gru_predictor, transformer_predictor

if __name__ == '__main__':
	lstm_model, gru_model, transformer_model = main()
