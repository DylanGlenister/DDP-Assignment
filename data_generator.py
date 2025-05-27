"""
    Used for generating realistic data to use for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
random_improvements = rng_gen.normal(loc=0.001, scale=0.0006, size=100)

base_day_offsets = rng_gen.uniform(low=0, high=365, size=100).astype(int).tolist()

base_day = datetime.fromisoformat("2023-01-01")

archer_list = {
    "ArcherID":         [],
    "Date":             [],
    "ScoreFraction":    []
}

for archer in range(100):
    archer_id = archer
    mean_skill = random_means[archer]
    improvement_factor = random_improvements[archer]
    initial_day = base_day + timedelta(days=base_day_offsets[archer])
    
    day_changes = rng_gen.uniform(low=1, high=7, size=100).astype(int).tolist()

    current_day = initial_day
    for day in range(100):
        archer_list["ArcherID"].append(archer_id)
        current_day = current_day + timedelta(days=day_changes[day])
        archer_list["Date"].append(current_day)
        archer_list["ScoreFraction"].append(rng_gen.normal(loc=mean_skill+(improvement_factor*day_changes[day]), scale=0.07, size=None))


df = pd.DataFrame(archer_list)
df.to_csv('data.csv', index=False)