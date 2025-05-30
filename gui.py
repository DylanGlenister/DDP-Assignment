"""
	Create a gui to use the program.

	When using database:
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
	|      Enter firstname:   [ENTRY BOX       ]             |
	|      Enter lastname:    [ENTRY BOX       ]             |
	|      Enter birthyear:   [ENTRY BOX       ]             |
	|      Select round:      [ROUND DROPDOWN  ]V            |
	|      Select model:      [MODEL DROPDOWN  ]V            |
	|  [Load previous scores] [Calculate score expectation]  |
	|                     Message                            |
	+--------------------------------------------------------+

	When using csv:
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
	|    Select archer:       [ARCHER DROPDOWN ]V            |
	|    Select model:        [MODEL DROPDOWN  ]V            |
	|  [Load previous scores] [Calculate score expectation]  |
	|                     Message                            |
	+--------------------------------------------------------+
"""

import sys
import tkinter as tk
import tkinter.ttk as ttk
from datetime import datetime

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import \
    NavigationToolbar2Tk  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import machine_learning as ml
import shared
from retrieve_db_data import DB_Retriever


class DataManager:
	"""Handles all management of data operations."""
	def __init__(self, _useDB: bool = True):
		self.usingDB=_useDB
		if self.usingDB:
			self.dbRetriever = DB_Retriever()
		else:
			self.data = pd.read_csv(shared.PATH_DATASET)
			self.data[shared.COLUMN_DATE] = pd.to_datetime(self.data[shared.COLUMN_DATE])
			self.archer_ids = sorted(self.data[shared.COLUMN_ARCHER_ID].unique())
			print(f'Loaded data for {len(self.archer_ids)} archers with {len(self.data)} total records')

	def get_round_info(self):
		"""Get the names of all of the available rounds."""
		if self.usingDB:
			return self.dbRetriever.get_round_info()
		else:
			# If the database isn't being used, return this cached version
			return pd.DataFrame({
				shared.COLUMN_ROUND_NAME: [
					'WA90/1440', 'WA70/1440', 'WA60/1440', 'AA50/1440', 'AA40/1440', 'WA70/720', 'WA60/720', 'WA50/720', 'Canberra', 'Long Sydney', 'Sydney', 'Long Brisbane', 'Brisbane', 'Adelaide', 'Short Adelaide', 'Hobart', 'Perth', 'Short Canberra', 'Junior Canberra', 'Mini Canberra', 'Grange', 'Melbourne', 'Darwin', 'Geelong', 'Newcastle', 'Holt', 'Samford', 'Drake', 'Wollongong', 'Townsville', 'Launceston', 'Full Spread',
				],
				shared.COLUMN_MAX_SCORE: [
					1440, 1440, 1440, 1440, 1080, 720, 720, 720, 900, 1200, 1200, 1200, 1200, 1200, 1200, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 720, 720, 720, 1200
				]
			})

	# === Fractional data from DB ===

	def get_fractional_scores_from_db(
		self,
		_firstname: str,
		_lastname: str,
		_birthyear: int,
		_round: str | None = None
	) -> pd.DataFrame:
		"""Get all data for a specific archer, sorted by date."""
		assert(self.usingDB)
		return self.dbRetriever.get_scores_as_fraction(
			_firstname,
			_lastname,
			_birthyear,
			_round
		).sort_values(shared.COLUMN_DATE)

	def get_recent_fractional_scores_from_db(
		self,
		_firstname: str,
		_lastname: str,
		_birthyear: int,
		_round: str | None = None,
		_sequence_length: int = 12
	) -> list[float]:
		"""Get the most recent scores for an archer."""
		archer_data = self.get_fractional_scores_from_db(
			_firstname,
			_lastname,
			_birthyear,
			_round
		)
		return archer_data[shared.COLUMN_SCORE].tail(_sequence_length).to_list()

	# === Data from CSV ===

	def get_fractional_scores_from_csv(self, _archer_id: int) -> pd.DataFrame:
		"""Get all data for a specific archer, sorted by date."""
		assert(not self.usingDB)
		return self.data[self.data[shared.COLUMN_ARCHER_ID] == _archer_id].sort_values(shared.COLUMN_DATE)

	def get_recent_fractional_scores_from_csv(
		self,
		_archer_id: int,
		_sequence_length: int = 12
	) -> list[float]:
		"""Get the most recent scores for an archer."""
		archer_data = self.get_fractional_scores_from_csv(_archer_id)
		return archer_data[shared.COLUMN_SCORE].tail(_sequence_length).tolist()


class ModelManager:
	"""Handles loading and managing prediction models."""

	def __init__(self):
		self.predictors: dict[str, ml.ArcheryPredictor] = {}
		self._load_all_models()

	def _load_single_model(self, _model_type: str) -> ml.ArcheryPredictor | None:
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

	def get_display_names(self) -> list[str]:
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

	def plot_historical_data(
		self,
		_archer_data: pd.DataFrame,
		_archer_label: str
	):
		"""Plot historical data for an archer."""
		self.clear_plot()
		plot = self.fig.add_subplot(111)

		plot.plot(
			_archer_data[shared.COLUMN_DATE],
			_archer_data[shared.COLUMN_SCORE],
			'o-',
			color='blue',
			label=f'Archer {_archer_label} - Historical',
			linewidth=2
		)

		self._setup_plot(plot, 'Archery Scores - Historical Data')
		self.canvas.draw()

	def plot_with_prediction(
		self,
		_archer_data: pd.DataFrame,
		_archer_id: int,
		_predicted_score: float | int,
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

		if isinstance(_predicted_score, float):
			# Annotation
			plot.annotate(
				f'Predicted: {_predicted_score:.4f}',
				xy=(current_date, _predicted_score),  # type: ignore
				xytext=(10, 10),
				textcoords='offset points',
				bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
				arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
			)
		else:
			# Annotation
			plot.annotate(
				f'Predicted: {_predicted_score}',
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
		_plot.set_xlabel(shared.COLUMN_DATE)
		_plot.set_ylabel('Score')
		_plot.set_title(_title)
		_plot.legend()
		_plot.grid(True, alpha=0.3)
		_plot.tick_params(axis='x', rotation=45)
		self.fig.tight_layout()


class ArcheryPredictionGUI:
	"""Main application class that coordinates all components."""

	def __init__(self, _useDB: bool = True):
		self.data_manager = DataManager(_useDB)
		self.model_manager = ModelManager()

		self.root = tk.Tk()
		self.root.title('Archery Score Prediction System')
		if _useDB:
			self.root.geometry('800x1000')
		else:
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
		graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

		fig = Figure(figsize=(8, 6), dpi=100)
		canvas = FigureCanvasTkAgg(fig, master=graph_frame)
		toolbar = NavigationToolbar2Tk(canvas, graph_frame)
		toolbar.update()
		canvas.get_tk_widget().pack()

		self.plot_manager = PlotManager(fig, canvas)

	def _setup_controls(self, _parent):
		"""Setup input controls and buttons."""
		input_frame = tk.Frame(_parent)
		input_frame.pack(fill=tk.X, pady=5)

		# Firstname entry box
		firstname_frame = tk.Frame(input_frame)
		firstname_frame.pack(pady=2)
		tk.Label(
			firstname_frame,
			text='Enter firstname:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.firstname_entry = tk.Entry(firstname_frame)
		self.firstname_entry.pack(side=tk.LEFT)
		self.firstname_entry.focus()

		# Lastname entry box
		lastname_frame = tk.Frame(input_frame)
		lastname_frame.pack(pady=2)
		tk.Label(
			lastname_frame,
			text='Enter lastname:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.lastname_entry = tk.Entry(lastname_frame)
		self.lastname_entry.pack(side=tk.LEFT)

		# Year of birth selection
		birthyear_frame = tk.Frame(input_frame)
		birthyear_frame.pack(pady=2)
		tk.Label(
			birthyear_frame,
			text='Enter birthyear:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.birthyear_combo = ttk.Combobox(
			birthyear_frame,
			values=[str(i) for i in range(1900, 2025)]
		)
		self.birthyear_combo.pack(side=tk.LEFT)

		# Round selection
		round_frame = tk.Frame(input_frame)
		round_frame.pack(pady=2)
		tk.Label(
			round_frame,
			text='Select round:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		round_values = self.data_manager.get_round_info()[shared.COLUMN_ROUND_NAME].to_list()
		round_values.insert(0, 'All')
		self.round_combo = ttk.Combobox(
			round_frame,
			values=round_values
		)
		self.round_combo.current(0)
		self.round_combo.pack(side=tk.LEFT)

		# Model selection
		model_frame = tk.Frame(input_frame)
		model_frame.pack(pady=2)
		tk.Label(
			model_frame,
			text='Select model:',
			anchor=tk.W, width=15
		).pack(side=tk.LEFT, padx=(20, 10))
		self.model_combo = ttk.Combobox(model_frame, values=self.model_manager.get_display_names())
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

	def _get_firstname(self) -> str:
		"""Get the value entered into the firstname textbox."""
		return self.firstname_entry.get()

	def _get_lastname(self) -> str:
		"""Get the value entered into the lastname textbox."""
		return self.lastname_entry.get()

	def _get_birthyear(self) -> int:
		"""Get the value entered into the year of birth textbox."""
		return int(self.birthyear_combo.get())

	def _get_round(self) -> str:
		"""Get the currently selected round."""
		return self.round_combo.get()

	def _get_selected_model(self) -> str:
		"""Get the currently selected model name."""
		return self.model_combo.get()

	def load_historical_data(self):
		"""Load and display historical scores for the selected archer."""
		self._update_status('Loading historical scores...')
		round: str | None = None if (temp_round := self._get_round()) == 'All' else temp_round

		try:
			if self.data_manager.usingDB:
				firstname = self._get_firstname()
				lastname = self._get_lastname()
				birthyear = self._get_birthyear()
				archer_data = self.data_manager.get_fractional_scores_from_db(
					firstname,
					lastname,
					birthyear,
					round
				)

				archer_id = int(archer_data[shared.COLUMN_ARCHER_ID].iloc[0])

			else:
				try:
					archer_id = int(self._get_firstname())
					archer_data = self.data_manager.get_fractional_scores_from_csv(archer_id)
				except ValueError as e:
					error_msg = f'Error parsing id: {e}'
					print(error_msg)
					self._update_status(error_msg, 'red')
					return

			if archer_data.empty:
				self._update_status(f'No data found for archer {archer_id}', 'red')
				return

			# If a round is selected, denormalise the data
			if round is not None:
				round_data = self.data_manager.get_round_info()
				max_score = round_data[round_data[shared.COLUMN_ROUND_NAME] == round][shared.COLUMN_MAX_SCORE].iloc[0]
				archer_data[shared.COLUMN_SCORE] = archer_data[shared.COLUMN_SCORE] * max_score

			self.plot_manager.plot_historical_data(
				archer_data,
				str(archer_id)
			)
			self._update_status('Historical scores loaded successfully', 'green')

		except ValueError as e:
			error_msg = f'Error loading historical data: incorrect data entered'
			print(error_msg)
			self._update_status(error_msg, 'red')

		except KeyError as e:
			error_msg = f'Error loading historical data: No data matching these details found'
			print(error_msg)
			self._update_status(error_msg, 'red')

		except Exception as e:
			error_msg = f'Error loading historical data: {e}'
			print(error_msg)
			self._update_status(error_msg, 'red')

	def calculate_prediction(self):
		"""Calculate and display prediction for the selected archer."""
		self._update_status('Calculating prediction...')
		round: str | None = None if (temp_round := self._get_round()) == 'All' else temp_round

		try:
			model_name = self._get_selected_model()
			if not model_name:
				self._update_status('No model selected', 'red')
				return

			if self.data_manager.usingDB:
				firstname = self._get_firstname()
				lastname = self._get_lastname()
				birthyear = self._get_birthyear()
				archer_data = self.data_manager.get_fractional_scores_from_db(
					firstname,
					lastname,
					birthyear,
					round
				)
				archer_id = int(archer_data[shared.COLUMN_ARCHER_ID].iloc[0])
				recent_scores = self.data_manager.get_recent_fractional_scores_from_db(
					firstname,
					lastname,
					birthyear,
					round
				)
			else:
				try:
					archer_id = int(self._get_firstname())
					archer_data = self.data_manager.get_fractional_scores_from_csv(archer_id)
					recent_scores = self.data_manager.get_recent_fractional_scores_from_csv(archer_id)
				except ValueError as e:
					error_msg = f'Error parsing id: {e}'
					print(error_msg)
					self._update_status(error_msg, 'red')
					return

			if not recent_scores:
				self._update_status(f'No score data available for archer {archer_id}', 'red')
				return

			predictor = self.model_manager.get_predictor(model_name)
			predicted_score = predictor.predict_score(archer_id, recent_scores)

			# If a round is selected, denormalise the data
			if round is not None:
				round_data = self.data_manager.get_round_info()
				max_score = round_data[round_data[shared.COLUMN_ROUND_NAME] == round][shared.COLUMN_MAX_SCORE].iloc[0]
				archer_data[shared.COLUMN_SCORE] = archer_data[shared.COLUMN_SCORE] * max_score
				recent_scores *= max_score
				predicted_score = int(predicted_score * max_score)

			# Update plot
			self.plot_manager.plot_with_prediction(
				archer_data,
				archer_id,
				predicted_score,
				model_name
			)

			success_msg = f'Prediction complete: {predicted_score:.4f} (using {model_name} model)'
			print(success_msg)
			self._update_status(success_msg, 'green')

		except ValueError as e:
			error_msg = f'Error calculating prediction: incorrect data entered'
			print(error_msg)
			self._update_status(error_msg, 'red')

		except KeyError as e:
			error_msg = f'Error calculating prediction: No data matching these details found'
			print(error_msg)
			self._update_status(error_msg, 'red')

		except Exception as e:
			error_msg = f'Error calculating prediction: {e}'
			print(error_msg)
			self._update_status(error_msg, 'red')

	def run(self):
		"""Start the application."""
		self.root.mainloop()


def main():
	"""Main entry point."""

	usedb = True
	if len(sys.argv) > 1 and sys.argv[1] == '--use_csv':
		usedb = False

	app = ArcheryPredictionGUI(usedb)
	app.run()


if __name__ == '__main__':
	main()
