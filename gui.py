'''
+======================================+
|               Title                  |
|                                      |
|  +--------------------------------+  |
|  |                                |  |
|  |   Use Matplotlib to display    |  |
|  |  a graph of their historical   |  |
|  |  scores and then a prediction  |  |
|  |     of their future score      |  |
|  |                                |  |
|  +--------------------------------+  |
|                                      |
|  [Load previous scores]              |
|  [Calculate score expectation]       |
|                                      |
+--------------------------------------+
'''

import tkinter as tk

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import \
    NavigationToolbar2Tk  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

# === BUTTONS ===
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=10)

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
load_btn.pack(pady=10)

def calculate():
	return

calculate_btn = tk.Button(button_frame, text='Calculate score expectation')
calculate_btn.pack(pady=10)

root.mainloop()
