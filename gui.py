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
|   Select model:         [MODEL DROPDOWN ]V             |
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

import machine_learning as ml

# === MAIN WINDOW ===
root = tk.Tk()
root.title('Title')
root.geometry('700x700')

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

# === MODEL DROPDOWN (First Row) ===
models = ['LSTM', 'GRU', 'TFMR']

model_frame = tk.Frame(input_frame)
model_frame.pack(pady=5)

label = tk.Label(
	model_frame,
	text='Select model:',
	anchor=tk.W
)
label.pack(side=tk.LEFT, padx=(0, 10))

model_combo = ttk.Combobox(model_frame, values=models)
model_combo.current(0)
model_combo.pack(side=tk.LEFT)

# === BUTTONS (Second Row) ===
button_frame = tk.Frame(input_frame)
button_frame.pack(pady=5)

def load():
	# list of squares
	y = [i**2 for i in range(101)]

	# adding the subplot
	plot1 = fig.add_subplot(111)

	# plotting the graph
	plot1.plot(y)

	canvas.draw()
	return

load_btn = tk.Button(button_frame, text='Load previous scores', command=load)
load_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

def calculate():
	return

calculate_btn = tk.Button(button_frame, text='Calculate score expectation')
calculate_btn.pack(side=tk.LEFT, pady=10)

# === RUN ===
root.mainloop()
