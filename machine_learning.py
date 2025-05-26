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

class TrafficDataset(Dataset):
	'''Custom dataset for traffic volume prediction'''

	def __init__(self, features, targets, scats_encoded, direction_encoded, time_encoded):
		self.features = torch.FloatTensor(features)
		self.targets = torch.FloatTensor(targets)
		self.scats_encoded = torch.LongTensor(scats_encoded)
		self.direction_encoded = torch.LongTensor(direction_encoded)
		self.time_encoded = torch.LongTensor(time_encoded)

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		return {
			'features': self.features[idx],
			'targets': self.targets[idx],
			'scats': self.scats_encoded[idx],
			'direction': self.direction_encoded[idx],
			'time': self.time_encoded[idx]
		}

class TrafficRNN(nn.Module):
	'''Unified RNN model supporting LSTM, GRU, and Transformer architectures'''

	def __init__(self, sequence_length, n_scats, n_directions, model_type='lstm',
				 hidden_dim=64, n_layers=2, dropout=0.2, n_heads=8):
		super(TrafficRNN, self).__init__()

		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.sequence_length = sequence_length
		self.model_type = model_type.lower()

		# Embedding layers for categorical features
		self.scats_embedding = nn.Embedding(n_scats, 16)
		self.direction_embedding = nn.Embedding(n_directions, 8)
		self.time_embedding = nn.Embedding(96, 16)  # 96 time slots (24h * 4 per hour)

		# RNN input size: volume + embeddings
		rnn_input_size = 1 + 16 + 8 + 16  # volume + scats_emb + direction_emb + time_emb = 41

		# Create the appropriate model type
		if self.model_type == 'lstm':
			self.rnn = nn.LSTM(rnn_input_size, hidden_dim, n_layers,
							  batch_first=True, dropout=dropout if n_layers > 1 else 0)
		elif self.model_type == 'gru':
			self.rnn = nn.GRU(rnn_input_size, hidden_dim, n_layers,
							 batch_first=True, dropout=dropout if n_layers > 1 else 0)
		elif self.model_type == 'transformer':
			# For transformer, we need d_model to be divisible by n_heads
			# Let's use hidden_dim as d_model and project the input to match
			d_model = hidden_dim

			# Ensure d_model is divisible by n_heads
			if d_model % n_heads != 0:
				# Adjust d_model to be divisible by n_heads
				d_model = ((d_model // n_heads) + 1) * n_heads
				print(f'Adjusted d_model to {d_model} to be divisible by {n_heads} heads')

			# Project input to d_model dimensions
			self.input_projection = nn.Linear(rnn_input_size, d_model)

			# Transformer encoder layer
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=d_model,
				nhead=n_heads,
				dim_feedforward=d_model * 2,
				dropout=dropout,
				batch_first=True
			)
			self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
			# Update hidden_dim to match d_model for consistency
			self.hidden_dim = d_model
		else:
			raise ValueError(f'model_type must be "lstm", "gru", or "transformer"')

		# Output layers
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(self.hidden_dim, 1)

	def forward(self, x, scats, direction, time):
		# Get embeddings
		scats_emb = self.scats_embedding(scats)  # (batch_size, 16)
		direction_emb = self.direction_embedding(direction)  # (batch_size, 8)
		time_emb = self.time_embedding(time)  # (batch_size, 16)

		# Expand embeddings to match sequence length
		scats_emb = scats_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)
		direction_emb = direction_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)
		time_emb = time_emb.unsqueeze(1).repeat(1, self.sequence_length, 1)

		# Concatenate volume sequence with embeddings
		rnn_input = torch.cat([x, scats_emb, direction_emb, time_emb], dim=2)

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

class TrafficPredictor:
	'''Main class for traffic volume prediction'''

	def __init__(self, sequence_length=12, model_type='lstm'):
		self.sequence_length = sequence_length
		self.model_type = model_type.lower()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Preprocessing objects
		self.sequence_scaler = StandardScaler()
		self.target_scaler = StandardScaler()
		self.scats_encoder = LabelEncoder()
		self.direction_encoder = LabelEncoder()

		# Model and training objects
		self.model = None
		self.optimizer = None
		self.criterion = nn.MSELoss()

		# Model dimensions (will be set during data preparation or loading)
		self.n_scats = None
		self.n_directions = None

		print(f'Using device: {self.device}')

	def load_and_preprocess_data(self, filepath=shared.PATH_DATASET):
		'''Load and preprocess the traffic data'''
		print('Loading data...')
		df = pd.read_csv(filepath)

		# Set multi-index
		df = df.set_index([shared.COLUMN_SCAT, shared.COLUMN_DIRECTION])

		# Get time columns (all columns except Latitude, Longitude)
		time_cols = [col for col in df.columns if col not in [shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE]]

		print(f'Data shape: {df.shape}')
		print(f'Number of time columns: {len(time_cols)}')

		return df, time_cols

	def create_sequences(self, df, time_cols):
		'''Create sequences for training'''
		print('Creating sequences...')

		sequences = []
		targets = []
		scats_list = []
		directions_list = []
		time_indices = []

		# Convert time columns to time indices (0-95 for 96 15-minute intervals)
		#time_to_idx = {col: idx for idx, col in enumerate(time_cols)}

		for (scats, direction), row in df.iterrows():
			# Get traffic volume values
			volumes = row[time_cols].values.astype(float)

			# Create sequences
			for i in range(len(volumes) - self.sequence_length):
				seq = volumes[i:i + self.sequence_length]
				target = volumes[i + self.sequence_length]

				sequences.append(seq)
				targets.append(target)
				scats_list.append(scats)
				directions_list.append(direction)
				time_indices.append(i + self.sequence_length)  # Target time index

		return np.array(sequences), np.array(targets), scats_list, directions_list, time_indices

	def prepare_data(self, filepath=shared.PATH_DATASET, test_size=0.2):
		'''Prepare data for training'''
		# Load data
		df, time_cols = self.load_and_preprocess_data(filepath)

		# Create sequences
		sequences, targets, scats_list, directions_list, time_indices = self.create_sequences(df, time_cols)

		# Encode categorical variables
		scats_encoded = self.scats_encoder.fit_transform(scats_list)
		directions_encoded = self.direction_encoder.fit_transform(directions_list)

		# Scale the sequences and targets separately
		sequences_scaled = self.sequence_scaler.fit_transform(sequences)
		targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

		# Split data
		X_train, X_test, y_train, y_test, scats_train, scats_test, dir_train, dir_test, time_train, time_test = train_test_split(
			sequences_scaled, targets_scaled, scats_encoded, directions_encoded, time_indices,
			test_size=test_size, random_state=42, stratify=scats_encoded
		)

		# Reshape for RNN input (batch_size, sequence_length, features)
		X_train = X_train.reshape(-1, self.sequence_length, 1)
		X_test = X_test.reshape(-1, self.sequence_length, 1)

		# Create datasets
		train_dataset = TrafficDataset(X_train, y_train, scats_train, dir_train, time_train)
		test_dataset = TrafficDataset(X_test, y_test, scats_test, dir_test, time_test)

		self.n_scats = len(self.scats_encoder.classes_)
		self.n_directions = len(self.direction_encoder.classes_)

		print(f'Training samples: {len(train_dataset)}')
		print(f'Test samples: {len(test_dataset)}')
		print(f'Number of SCATS: {self.n_scats}')
		print(f'Number of directions: {self.n_directions}')

		return train_dataset, test_dataset

	def create_model(self, hidden_dim=64, n_layers=2, dropout=0.2, n_heads=8):
		'''Create LSTM, GRU, or Transformer model'''
		self.model = TrafficRNN(
			self.sequence_length, self.n_scats, self.n_directions,
			self.model_type, hidden_dim, n_layers, dropout, n_heads
		)

		self.model = self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		print(f'Created {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters')

	def train(self, train_dataset, test_dataset, epochs=100, batch_size=64):
		'''Train the model'''
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		train_losses: list[float] = []
		test_losses: list[float] = []

		if self.model is None:
			raise ValueError('Cannot train, model as not been set')

		if self.optimizer is None:
			raise ValueError('Cannot train, optimiser as not been set')

		print(f'Training {self.model_type.upper()} model...')

		for epoch in range(epochs):
			# Training
			self.model.train()
			train_loss: float = 0

			for batch in train_loader:
				features = batch['features'].to(self.device)
				targets = batch['targets'].to(self.device)
				scats = batch['scats'].to(self.device)
				direction = batch['direction'].to(self.device)
				time = batch['time'].to(self.device)

				self.optimizer.zero_grad()
				outputs = self.model(features, scats, direction, time)
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
					scats = batch['scats'].to(self.device)
					direction = batch['direction'].to(self.device)
					time = batch['time'].to(self.device)

					outputs = self.model(features, scats, direction, time)
					loss = self.criterion(outputs.squeeze(), targets)
					test_loss += loss.item()

			train_loss /= len(train_loader)
			test_loss /= len(test_loader)

			train_losses.append(train_loss)
			test_losses.append(test_loss)

			if (epoch + 1) % 10 == 0:
				print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

		return train_losses, test_losses

	def predict_volume(self, scats, direction, time_str, recent_volumes=None):
		'''
		Predict traffic volume for given SCATS, direction, and time

		Args:
			scats: SCATS identifier
			direction: Direction (e.g., 'North', 'South', etc.)
			time_str: Time string in format 'HH:MM' (e.g., '08:30')
			recent_volumes: List of recent volume values for sequence (optional)

		Returns:
			Predicted traffic volume
		'''
		if self.model is None:
			raise ValueError('Cannot predict, model has not been set')

		self.model.eval()

		# Convert time string to time index
		hour, minute = map(int, time_str.split(':'))
		time_idx = hour * 4 + minute // 15  # Convert to 15-minute interval index

		# Encode categorical variables
		try:
			# Ignore these errors
			scats_encoded = self.scats_encoder.transform([scats])[0]
			direction_encoded = self.direction_encoder.transform([direction])[0]
		except ValueError as e:
			raise ValueError(f'Unknown SCATS or Direction: {e}')

		# If recent_volumes not provided, use dummy sequence (this is not ideal for real prediction)
		if recent_volumes is None:
			print('Warning: No recent volume data provided. Using dummy sequence.')
			recent_volumes = [0.0] * self.sequence_length

		if len(recent_volumes) < self.sequence_length:
			# Pad with zeros if not enough data
			recent_volumes = [0.0] * (self.sequence_length - len(recent_volumes)) + list(recent_volumes)
		elif len(recent_volumes) > self.sequence_length:
			# Take the last sequence_length values
			recent_volumes = recent_volumes[-self.sequence_length:]

		# Scale the input sequence using sequence_scaler
		sequence_scaled = self.sequence_scaler.transform([recent_volumes])

		# Prepare input tensors
		features = torch.FloatTensor(sequence_scaled).reshape(1, self.sequence_length, 1).to(self.device)
		scats_tensor = torch.LongTensor([scats_encoded]).to(self.device)
		direction_tensor = torch.LongTensor([direction_encoded]).to(self.device)
		time_tensor = torch.LongTensor([time_idx]).to(self.device)

		# Make prediction
		with torch.no_grad():
			prediction_scaled = self.model(features, scats_tensor, direction_tensor, time_tensor)

		# Inverse transform using target_scaler to get actual volume
		prediction = self.target_scaler.inverse_transform([[prediction_scaled.cpu().item()]])[0][0]

		return max(0, prediction)  # Ensure non-negative volume

	def save_model(self, filepath=None):
		'''Save the trained model and preprocessors'''
		if filepath is None:
			filepath = f'traffic_{self.model_type}_model.pt'

		if self.model is None:
			raise ValueError('No model to save. Train a model first.')

		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
			'sequence_scaler': self.sequence_scaler,
			'target_scaler': self.target_scaler,
			'scats_encoder': self.scats_encoder,
			'direction_encoder': self.direction_encoder,
			'sequence_length': self.sequence_length,
			'model_type': self.model_type,
			'n_scats': self.n_scats,
			'n_directions': self.n_directions,
			'model_params': {
				'hidden_dim': self.model.hidden_dim,
				'n_layers': self.model.n_layers,
				'n_heads': getattr(self.model, 'n_heads', 8)  # Default for non-transformer models
			}
		}

		torch.save(checkpoint, filepath)
		print(f'Model saved to {filepath}')

	def load_model(self, filepath=None):
		'''Load a trained model and preprocessors'''
		if filepath is None:
			filepath = f'traffic_{self.model_type}_model.pt'

		try:
			checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

			# Load preprocessors
			self.sequence_scaler = checkpoint['sequence_scaler']
			self.target_scaler = checkpoint['target_scaler']
			self.scats_encoder = checkpoint['scats_encoder']
			self.direction_encoder = checkpoint['direction_encoder']
			self.sequence_length = checkpoint['sequence_length']
			self.n_scats = checkpoint['n_scats']
			self.n_directions = checkpoint['n_directions']

			# Recreate model with saved parameters
			model_params = checkpoint['model_params']
			# Model will not be None after this
			self.create_model(
				hidden_dim=model_params['hidden_dim'],
				n_layers=model_params['n_layers'],
				n_heads=model_params.get('n_heads', 8)  # Default for backward compatibility
			)
			assert(self.model is not None)

			# Load model state
			self.model.load_state_dict(checkpoint['model_state_dict'])

			# Load optimizer state if available
			if checkpoint['optimizer_state_dict'] is not None and self.optimizer is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

			print(f'Model loaded from {filepath}')
			return True

		except FileNotFoundError:
			print(f'Model file {filepath} not found.')
			return False
		except Exception as e:
			print(f'Error loading model: {e}')
			return False

	def plot_training_history(self, train_losses, test_losses):
		'''Plot training history'''
		plt.figure(figsize=(10, 6))
		plt.plot(train_losses, label='Training Loss')
		plt.plot(test_losses, label='Validation Loss')
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

		predictor = TrafficPredictor(sequence_length=12, model_type=model_type)

		# Try to load existing model
		model_loaded = predictor.load_model()

		if model_loaded:
			print(f'✓ Successfully loaded existing {model_type.upper()} model')

			# Verify model is working by checking if we can make a dummy prediction
			try:
				# Test with dummy data - in practice you'd use real values
				if hasattr(predictor, 'n_scats') and predictor.n_scats is not None:
					print(f'Model ready for predictions. SCATS count: {predictor.n_scats}, Directions: {predictor.n_directions}')
				else:
					print('Warning: Model loaded but some parameters may be missing')
			except Exception as e:
				print(f'Warning: Loaded model may have issues: {e}')

		else:
			print(f'✗ No existing {model_type.upper()} model found. Training new model...')

			try:
				# Prepare data
				train_dataset, test_dataset = predictor.prepare_data('processed.csv')

				# Create and train model
				predictor.create_model(hidden_dim=64, n_layers=2, dropout=0.2)
				print(f'Training {model_type.upper()} model...')
				assert(predictor is not None)
				train_losses, test_losses = predictor.train(
					train_dataset, test_dataset,
					epochs=50, batch_size=64
				)

				# Save the trained model
				predictor.save_model()

				# Plot training history
				predictor.plot_training_history(train_losses, test_losses)

				print(f'✓ {model_type.upper()} model trained and saved successfully')

			except FileNotFoundError:
				print(f'✗ Error: 'processed.csv' not found. Please ensure the data file exists.')
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
		print('predictor.predict_volume(scats, direction, time_str, recent_volumes)')
		print('\nExample:')
		print('recent_volumes = [100, 120, 110, 95, 80, 70, 85, 90, 105, 115, 125, 130]')
		print('prediction = predictor.predict_volume("your_scats", "your_direction", "08:30", recent_volumes)')

		# Show available SCATS and directions if models are loaded
		try:
			# Ignore these errors
			if hasattr(lstm_predictor, 'scats_encoder') and hasattr(lstm_predictor.scats_encoder, 'classes_'):
				print(f'\nAvailable SCATS ({len(lstm_predictor.scats_encoder.classes_)}): {list(lstm_predictor.scats_encoder.classes_[:5])}{'...' if len(lstm_predictor.scats_encoder.classes_) > 5 else ''}')
				print(f'Available Directions: {list(lstm_predictor.direction_encoder.classes_)}')
		except:
			pass

	return lstm_predictor, gru_predictor, transformer_predictor

if __name__ == '__main__':
	lstm_model, gru_model, transformer_model = main()
