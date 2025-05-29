"""
+========================================================+
|                        Title                           |
|                                                        |
|  +--------------------------------------------------+  |
|  |                                                  |  |
|  |      Use Matplotlib to display a graph of        |  |
|  |       their historical scores and then a         |  |
|  |        prediction of their future score          |  |
|  |                                                  |  |
|  +--------------------------------------------------+  |
|                                                        |
|   Select archer:        [ARCHER DROPDOWN ]V            |
|   Select model:         [MODEL DROPDOWN  ]V            |
|  [Load previous scores] [Calculate score expectation]  |
|                                                        |
+--------------------------------------------------------+
"""

import tkinter as tk
import tkinter.ttk as ttk
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import \
	NavigationToolbar2Tk  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import machine_learning as ml
import shared


class DataManager:
	"""Handles all data loading and processing operations."""

	def __init__(self, _csv_path: str):
		self.data = pd.read_csv(_csv_path)
		self.data[shared.COLUMN_DATE] = pd.to_datetime(self.data[shared.COLUMN_DATE])
		self.archer_ids = sorted(self.data[shared.COLUMN_ARCHER_ID].unique())
		print(f'Loaded data for {len(self.archer_ids)} archers with {len(self.data)} total records')

	def get_archer_data(self, _archer_id: int) -> pd.DataFrame:
		"""Get all data for a specific archer, sorted by date."""
		return self.data[self.data[shared.COLUMN_ARCHER_ID] == _archer_id].sort_values(shared.COLUMN_DATE)

	def get_recent_scores(self, _archer_id: int, _sequence_length: int = 12) -> List[float]:
		"""Get the most recent scores for an archer."""
		archer_data = self.get_archer_data(_archer_id)
		return archer_data[shared.COLUMN_SCORE].tail(_sequence_length).tolist()


class ModelManager:
	"""Handles loading and managing prediction models."""

	def __init__(self):
		self.predictors: Dict[str, ml.ArcheryPredictor] = {}
		self._load_all_models()

	def _load_single_model(self, _model_type: str) -> Optional[ml.ArcheryPredictor]:
		"""Load a single model of the specified type."""
		predictor = ml.ArcheryPredictor(_sequence_length=12, _model_type=_model_type)
		if predictor.load_or_train():
			return predictor
		else:
			print(f'Failed to load {_model_type} model')
			return

	def _load_all_models(self):
		"""Load all available models."""
		print('Loading prediction models...')
		for model_type in shared.MODEL_TYPES:
			print(f'Loading {shared.MODEL_DISPLAY_NAMES[model_type]} model...')
			predictor = self._load_single_model(model_type)
			if predictor is not None:
				self.predictors[model_type] = predictor

	def get_predictor(self, _model_display_name: str) -> ml.ArcheryPredictor:
		"""Get predictor by display name (e.g., 'lstm' -> lstm predictor)."""
		model_type = next(k for k, v in shared.MODEL_DISPLAY_NAMES.items() if v == _model_display_name)
		predictor = self.predictors[model_type]
		if predictor is None:
			raise ValueError(f'Model {_model_display_name} not available')
		return predictor

	def get_display_names(self) -> List[str]:
		"""Get list of available model display names."""
		return [name for key, name in shared.MODEL_DISPLAY_NAMES.items() if self.predictors[key] is not None]

	@property
	def all_loaded(self) -> bool:
		"""Check if all models are loaded successfully."""
		return all(predictor is not None for predictor in self.predictors.values())


class PlotManager:
	"""Handles all plotting operations."""

	def __init__(self, _figure: Figure, _canvas: FigureCanvasTkAgg):
		self.fig = _figure
		self.canvas = _canvas

	def clear_plot(self):
		"""Clear the current plot."""
		self.fig.clear()

	def plot_historical_data(self, _archer_data: pd.DataFrame, _archer_id: int):
		"""Plot historical data for an archer."""
		self.clear_plot()
		plot = self.fig.add_subplot(111)

		plot.plot(
			_archer_data[shared.COLUMN_DATE],
			_archer_data[shared.COLUMN_SCORE],
			'o-',
			color='blue',
			label=f'Archer {_archer_id} - Historical',
			linewidth=2
		)

		self._setup_plot(plot, 'Archery Scores - Historical Data')
		self.canvas.draw()

	def plot_with_prediction(
			self,
			_archer_data: pd.DataFrame,
			_archer_id: int,
			_predicted_score: float,
			_model_name: str
		):
		"""Plot historical data with prediction."""
		self.clear_plot()
		plot = self.fig.add_subplot(111)

		# Historical data
		plot.plot(
			_archer_data[shared.COLUMN_DATE],
			_archer_data[shared.COLUMN_SCORE],
			'o-',
			color='blue',
			label=f'Archer {_archer_id} - Historical',
			linewidth=2
		)

		# Prediction
		current_date = datetime.now()
		plot.plot(
			[current_date],  # type: ignore
			[_predicted_score],
			'o',
			color='red',
			markersize=12,
			label=f'Predicted Score ({_model_name})',
			zorder=5
		)

		# Connection line
		last_date = _archer_data[shared.COLUMN_DATE].iloc[-1]
		last_score = _archer_data[shared.COLUMN_SCORE].iloc[-1]
		plot.plot(
			[last_date, current_date],  # type: ignore
			[last_score,
			_predicted_score],
			'--',
			color='red',
			alpha=0.7,
			linewidth=2
		)

		# Annotation
		plot.annotate(
			f'Predicted: {_predicted_score:.4f}',
			xy=(current_date, _predicted_score),  # type: ignore
			xytext=(10, 10),
			textcoords='offset points',
			bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
			arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
		)

		title = f'Archery Scores - Historical Data + {_model_name} Prediction'
		self._setup_plot(plot, title)
		self.canvas.draw()

	def _setup_plot(self, _plot, _title: str):
		"""Common plot setup operations."""
		_plot.set_xlabel('Date')
		_plot.set_ylabel('Score Fraction')
		_plot.set_title(_title)
		_plot.legend()
		_plot.grid(True, alpha=0.3)
		_plot.tick_params(axis='x', rotation=45)
		self.fig.tight_layout()


class ArcheryPredictionGUI:
	"""Main application class that coordinates all components."""

	def __init__(self):
		self.data_manager = DataManager(shared.PATH_DATASET)
		self.model_manager = ModelManager()

		self.root = tk.Tk()
		self.root.title('Archery Score Prediction System')
		self.root.geometry('800x900')

		self._setup_ui()
		self._update_initial_status()

	def _setup_ui(self):
		"""Setup the user interface."""
		main_frame = tk.Frame(self.root)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		# Graph setup
		self._setup_graph(main_frame)

		# Input controls setup
		self._setup_controls(main_frame)

	def _setup_graph(self, _parent):
		"""Setup the matplotlib graph area."""
		graph_frame = tk.Frame(_parent)
		graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

		fig = Figure(figsize=(8, 6), dpi=100)
		canvas = FigureCanvasTkAgg(fig, master=graph_frame)
		toolbar = NavigationToolbar2Tk(canvas, graph_frame)
		toolbar.update()
		canvas.get_tk_widget().pack()

		self.plot_manager = PlotManager(fig, canvas)

	def _setup_controls(self, _parent):
		"""Setup input controls and buttons."""
		input_frame = tk.Frame(_parent)
		input_frame.pack(fill=tk.X, pady=10)

		# Archer selection
		archer_frame = tk.Frame(input_frame)
		archer_frame.pack(pady=5)

		tk.Label(
			archer_frame,
			text='Select archer:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.archer_combo = ttk.Combobox(archer_frame, values=self.data_manager.archer_ids)
		if self.data_manager.archer_ids:
			self.archer_combo.current(0)
		self.archer_combo.pack(side=tk.LEFT)

		# Model selection
		model_frame = tk.Frame(input_frame)
		model_frame.pack(pady=5)

		tk.Label(
			model_frame,
			text='Select model:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.model_combo = ttk.Combobox(model_frame, values=self.model_manager.get_display_names())
		available_models = self.model_manager.get_display_names()
		if available_models:
			self.model_combo.current(0)
		self.model_combo.pack(side=tk.LEFT)

		# Buttons
		button_frame = tk.Frame(input_frame)
		button_frame.pack(pady=5)

		tk.Button(
			button_frame,
			text='Load previous scores',
			command=self.load_historical_data
		).pack(side=tk.LEFT, padx=(0, 10), pady=10)
		tk.Button(
			button_frame,
			text='Calculate score expectation',
			command=self.calculate_prediction
		).pack(side=tk.LEFT, pady=10)

		# Status label
		status_frame = tk.Frame(input_frame)
		status_frame.pack(pady=5)
		self.status_label = tk.Label(status_frame, text='', fg='blue')
		self.status_label.pack()

	def _update_status(self, _message: str, _color: str = 'blue'):
		"""Update the status message."""
		self.status_label.config(text=_message, fg=_color)
		self.root.update()

	def _update_initial_status(self):
		"""Set the initial status message."""
		if self.model_manager.all_loaded:
			self._update_status('Models loaded successfully. Ready for predictions.', 'green')
		else:
			loaded_models = [name for name in self.model_manager.get_display_names()]
			if loaded_models:
				self._update_status(f'Partial load: {", ".join(loaded_models)} available.', 'orange')
			else:
				self._update_status('Warning: No models loaded. Please check model files.', 'red')

	def _get_selected_archer(self) -> int:
		"""Get the currently selected archer ID."""
		return int(self.archer_combo.get())

	def _get_selected_model(self) -> str:
		"""Get the currently selected model name."""
		return self.model_combo.get()

	def load_historical_data(self):
		"""Load and display historical scores for the selected archer."""
		try:
			self._update_status('Loading historical scores...')

			archer_id = self._get_selected_archer()
			archer_data = self.data_manager.get_archer_data(archer_id)

			if archer_data.empty:
				self._update_status(f'No data found for archer {archer_id}', 'red')
				return

			self.plot_manager.plot_historical_data(archer_data, archer_id)
			self._update_status('Historical scores loaded successfully', 'green')

		except Exception as e:
			error_msg = f'Error loading historical data: {e}'
			print(error_msg)
			self._update_status(error_msg, 'red')

	def calculate_prediction(self):
		"""Calculate and display prediction for the selected archer."""
		try:
			self._update_status('Calculating prediction...')

			archer_id = self._get_selected_archer()
			model_name = self._get_selected_model()

			if not model_name:
				self._update_status('No model selected', 'red')
				return

			# Get data and make prediction
			recent_scores = self.data_manager.get_recent_scores(archer_id)

			if not recent_scores:
				self._update_status(f'No score data available for archer {archer_id}', 'red')
				return

			predictor = self.model_manager.get_predictor(model_name)
			predicted_score = predictor.predict_score(archer_id, recent_scores)

			# Update plot
			archer_data = self.data_manager.get_archer_data(archer_id)
			self.plot_manager.plot_with_prediction(archer_data, archer_id, predicted_score, model_name)

			success_msg = f'Prediction complete: {predicted_score:.4f} (using {model_name} model)'
			print(success_msg)
			self._update_status(success_msg, 'green')

		except Exception as e:
			error_msg = f'Error calculating prediction: {e}'
			print(error_msg)
			self._update_status(error_msg, 'red')

	def run(self):
		"""Start the application."""
		self.root.mainloop()


def main():
	"""Main entry point."""
	app = ArcheryPredictionGUI()
	app.run()


if __name__ == '__main__':
	main()
