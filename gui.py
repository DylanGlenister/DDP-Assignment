'''
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
'''

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

# Load and process CSV data at startup
data = pd.read_csv(shared.PATH_DATASET)
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
# Get unique archer IDs
archer_ids = sorted(data['ArcherID'].unique())
print(f'Loaded data for {len(archer_ids)} archers with {len(data)} total records')

# Initialize model predictors
lstm_predictor = None
gru_predictor = None
transformer_predictor = None

def load_model(_type: str):
	path = f'{shared.PATH_MODELS}archery_{_type}_model.pt'
	predictor = ml.ArcheryPredictor(_sequence_length=12, _model_type=_type)
	success = predictor.load_model(path)
	return predictor if success else None

def load_models():
	'''Load the pre-trained models'''
	global lstm_predictor, gru_predictor, transformer_predictor

	# Load LSTM model
	lstm_predictor = load_model('lstm')

	# Load GRU model
	gru_predictor = load_model('gru')

	# Load Transformer model
	transformer_predictor = load_model('transformer')

	return not any(model is None for model in [lstm_predictor, gru_predictor, transformer_predictor])

# Load models at startup
all_models_loaded = load_models()

# === MAIN WINDOW ===
root = tk.Tk()
root.title('Archery Score Prediction System')
root.geometry('800x900')

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# === GRAPH ===
graph_frame = tk.Frame(main_frame)
graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# the figure that will contain the plot
fig = Figure(figsize=(8, 6), dpi=100)

# creating the Tkinter canvas
# containing the Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=graph_frame)

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas, graph_frame)
toolbar.update()

# placing the canvas on the Tkinter window
# placing the toolbar on the Tkinter window
canvas.get_tk_widget().pack()

# === USER INPUT ===
input_frame = tk.Frame(main_frame)
input_frame.pack(fill=tk.X, pady=10)

# === ARCHER DROPDOWN (First Row) ===
archer_frame = tk.Frame(input_frame)
archer_frame.pack(pady=5)

archer_label = tk.Label(
	archer_frame,
	text='Select archer:',
	anchor=tk.W,
	width=15
)
archer_label.pack(side=tk.LEFT, padx=(20, 10))

archer_combo = ttk.Combobox(archer_frame, values=archer_ids)
if archer_ids:
	archer_combo.current(0)
archer_combo.pack(side=tk.LEFT)

# === MODEL DROPDOWN (Second Row) ===
models = ['LSTM', 'GRU', 'TFMR']

model_frame = tk.Frame(input_frame)
model_frame.pack(pady=5)

model_label = tk.Label(
	model_frame,
	text='Select model:',
	anchor=tk.W,
	width=15
)
model_label.pack(side=tk.LEFT, padx=(20, 10))

model_combo = ttk.Combobox(model_frame, values=models)
model_combo.current(0)
model_combo.pack(side=tk.LEFT)

# === BUTTONS (Third Row) ===
button_frame = tk.Frame(input_frame)
button_frame.pack(pady=5)

# === STATUS LABEL ===
status_frame = tk.Frame(input_frame)
status_frame.pack(pady=5)

status_label = tk.Label(status_frame, text='', fg='blue')
status_label.pack()

def update_status(message, color='blue'):
	'''Update the status label'''
	status_label.config(text=message, fg=color)
	root.update()

def get_recent_scores(archer_id, sequence_length=12):
	'''Get the most recent scores for an archer'''
	global data

	archer_data = data[data['ArcherID'] == archer_id].sort_values('Date')
	recent_scores = archer_data['ScoreFraction'].tail(sequence_length).tolist()

	return recent_scores

def load():
	global data

	update_status('Loading historical scores...')

	# Clear previous plot
	fig.clear()

	# Plot data for selected archer
	plot1 = fig.add_subplot(111)

	selected_archer_str = archer_combo.get()
	if selected_archer_str:
		selected_archer = int(selected_archer_str)
		archer_data = data[data['ArcherID'] == selected_archer].sort_values('Date')
		plot1.plot(archer_data['Date'], archer_data['ScoreFraction'], 'o-',
					color='blue', label=f'Archer {selected_archer} - Historical')
	else:
		# Plot for first archer if none selected
		first_archer = archer_ids[0] if archer_ids else 0
		archer_data = data[data['ArcherID'] == first_archer].sort_values('Date')
		plot1.plot(archer_data['Date'], archer_data['ScoreFraction'], 'o-',
					color='blue', label=f'Archer {first_archer} - Historical')

	plot1.set_xlabel('Date')
	plot1.set_ylabel('Score Fraction')
	plot1.set_title('Archery Scores - Historical Data')
	plot1.legend()
	plot1.grid(True, alpha=0.3)

	# Rotate x-axis labels for better readability
	plot1.tick_params(axis='x', rotation=45)

	fig.tight_layout()
	canvas.draw()

	update_status('Historical scores loaded successfully', 'green')

def calculate():
	global data, lstm_predictor, gru_predictor, transformer_predictor

	selected_archer_str = archer_combo.get()
	selected_model = model_combo.get()

	update_status('Calculating prediction...')

	selected_archer = int(selected_archer_str)

	# Get recent scores for the selected archer
	recent_scores = get_recent_scores(selected_archer)

	# Select the appropriate model
	predictor = None
	if selected_model == 'LSTM':
		predictor = lstm_predictor
	elif selected_model == 'GRU':
		predictor = gru_predictor
	elif selected_model == 'TFMR':
		predictor = transformer_predictor

	if predictor is None:
		error_msg = f'Model {selected_model} not available'
		update_status(error_msg, 'red')
		return

	try:
		# Make prediction
		predicted_score = predictor.predict_score(selected_archer, recent_scores)
	except Exception as e:
		error_msg = f'Error calculating prediction: {e}'
		print(error_msg)
		update_status(error_msg, 'red')
		raise e

	# Clear previous plot and redraw with prediction
	fig.clear()
	plot1 = fig.add_subplot(111)

	# Plot historical data
	archer_data = data[data['ArcherID'] == selected_archer].sort_values('Date')
	plot1.plot(
		archer_data['Date'],
		archer_data['ScoreFraction'],
		'o-',
		color='blue',
		label=f'Archer {selected_archer} - Historical',
		linewidth=2
	)

	# Plot prediction point
	current_date = datetime.now()
	plot1.plot(
		[current_date],
		[predicted_score],
		'o',
		color='red',
		markersize=12,
		label=f'Predicted Score ({selected_model})',
		zorder=5
	)

	# Add a dashed line connecting the last historical point to the prediction
	last_date = archer_data['Date'].iloc[-1]
	last_score = archer_data['ScoreFraction'].iloc[-1]
	plot1.plot(
		[last_date, current_date],
		[last_score, predicted_score],
		'--',
		color='red',
		alpha=0.7,
		linewidth=2
	)

	# Add text annotation for the predicted value
	plot1.annotate(
		f'Predicted: {predicted_score:.4f}',
		xy=(current_date, predicted_score),
		xytext=(10, 10),
		textcoords='offset points',
		bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
		arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
	)

	plot1.set_xlabel('Date')
	plot1.set_ylabel('Score Fraction')
	plot1.set_title(f'Archery Scores - Historical Data + {selected_model} Prediction')
	plot1.legend()
	plot1.grid(True, alpha=0.3)

	# Rotate x-axis labels for better readability
	plot1.tick_params(axis='x', rotation=45)

	fig.tight_layout()
	canvas.draw()

	success_msg = f'Prediction complete: {predicted_score:.4f} (using {selected_model} model)'
	print(success_msg)
	update_status(success_msg, 'green')

load_btn = tk.Button(button_frame, text='Load previous scores', command=load)
load_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

calculate_btn = tk.Button(button_frame, text='Calculate score expectation', command=calculate)
calculate_btn.pack(side=tk.LEFT, pady=10)

# === INITIAL STATUS ===
if all_models_loaded:
	update_status('Models loaded successfully. Ready for predictions.', 'green')
else:
	update_status('Warning: Not all models not loaded. Predictions may be unavailable.', 'orange')

# === RUN ===
if __name__ == '__main__':
	root.mainloop()
