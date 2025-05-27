"""
    Used for generating realistic data to use for training
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Loop through all 100 archer IDs
# Generate distribution across 100 dates or whatever

# For generating training data:
# Each archer has a initial base level
# Each archer has a improvement rate
# Improvement is per day

# Need to rng select an initial day and number of days for each interval.

# Selects initial score means and improvements
rng_gen = np.random.default_rng()

random_means = rng_gen.normal(loc=0.6, scale=0.1, size=100)
random_improvements = rng_gen.uniform(low=0.0001, high=0.002, size=100)

archer_list = []

for archer in range(100): 
    archer_id = archer
    mean_skill = random_means[archer]
    improvement_factor = random_improvements[archer]

    for day in range(100):
        continue
