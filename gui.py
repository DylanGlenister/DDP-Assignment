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

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import \
	NavigationToolbar2Tk  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#import machine_learning as ml

PATH_DATASET = './data.csv'

# Load and process CSV data at startup
try:
	data = pd.read_csv(PATH_DATASET)
	# Convert Date column to datetime
	data['Date'] = pd.to_datetime(data['Date'])
	# Get unique archer IDs
	archer_ids = sorted(data['ArcherID'].unique())
	print(f"Loaded data for {len(archer_ids)} archers with {len(data)} total records")
except Exception as e:
	print(f"Error loading data: {e}")
	data = None
	archer_ids = []

# === MAIN WINDOW ===
root = tk.Tk()
root.title('Title')
root.geometry('700x800')

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# === GRAPH ===
graph_frame = tk.Frame(main_frame)
graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# the figure that will contain the plot
fig = Figure(figsize = (5, 5), dpi = 100)

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
archer_combo.pack(side=tk.LEFT)

model_combo.pack(side=tk.LEFT)
button_frame = tk.Frame(input_frame)
button_frame.pack(pady=5)

# === BUTTONS (Third Row) ===
def load():
	global data
	if data is None:
		raise ValueError('Calculate cannot be run as data has failed to load')

	try:
		# Clear previous plot
		fig.clear()

		# Plot data for selected archer
		plot1 = fig.add_subplot(111)

		selected_archer_str = archer_combo.get()
		if selected_archer_str:
			selected_archer = int(selected_archer_str)
			archer_data = data[data['ArcherID'] == selected_archer].sort_values('Date')
			plot1.plot(archer_data['Date'], archer_data['ScoreFraction'], 'o-', label=f'Archer {selected_archer}')
		else:
			# Plot for first archer if none selected
			first_archer = archer_ids[0] if archer_ids else 0
			archer_data = data[data['ArcherID'] == first_archer].sort_values('Date')
			plot1.plot(archer_data['Date'], archer_data['ScoreFraction'], 'o-', label=f'Archer {first_archer}')

		plot1.set_xlabel('Date')
		plot1.set_ylabel('Score Fraction')
		plot1.set_title('Historical Scores')
		plot1.legend()
		plot1.grid(True)

		# Rotate x-axis labels for better readability
		plot1.tick_params(axis='x', rotation=45)

		fig.tight_layout()
		canvas.draw()

	except Exception as e:
		print(f"Error plotting data: {e}")

	return

load_btn = tk.Button(button_frame, text='Load previous scores', command=load)
load_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

def calculate():
	global data
	if data is None:
		raise ValueError('Calculate cannot be run as data has failed to load')

	selected_archer = archer_combo.get()
	selected_model = model_combo.get()

	if not selected_archer:
		print("Please select an archer")
		return

	print(f"Calculate prediction for Archer {selected_archer} using {selected_model} model")
	# TODO: Implement prediction logic using machine learning
	return

calculate_btn = tk.Button(button_frame, text='Calculate score expectation')
calculate_btn.pack(side=tk.LEFT, pady=10)

# === RUN ===
root.mainloop()
