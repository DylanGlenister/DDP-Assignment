"""
	Used for generating realistic data to use for training
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import shared

# Loop through all 100 archer IDs
# Generate distribution across 100 dates or whatever

# For generating training data:
# Each archer has a initial base level
# Each archer has a improvement rate
# Improvement is per day

# Need to rng select an initial day and number of days for each interval.

# Selects initial score means and improvements
rng_gen = np.random.default_rng()

ARCHER_NUM = 100
ENTRIES_PER_ARCHER = 100

random_means = rng_gen.normal(loc=0.8, scale=0.1, size=ARCHER_NUM)
random_improvements = rng_gen.normal(loc=0.0008, scale=0.0003, size=ARCHER_NUM)

base_day_offsets = rng_gen.uniform(low=0, high=365, size=ARCHER_NUM).astype(int).tolist()

base_day = datetime.fromisoformat("2025-05-27")

archer_list = {
	shared.COLUMN_ARCHER_ID: [],
	shared.COLUMN_DATE:      [],
	shared.COLUMN_SCORE:     []
}

for archer in range(ARCHER_NUM):
	archer_id = archer
	mean_skill = random_means[archer]
	improvement_factor = random_improvements[archer]
	initial_day = base_day - timedelta(days=base_day_offsets[archer])

	day_changes = rng_gen.uniform(low=1, high=7, size=ENTRIES_PER_ARCHER).astype(int).tolist()

	current_day = initial_day
	current_skill = mean_skill
	for day in range(ENTRIES_PER_ARCHER):
		archer_list[shared.COLUMN_ARCHER_ID].append(archer_id)
		current_day = current_day - timedelta(days=day_changes[day])
		archer_list[shared.COLUMN_DATE].append(current_day)
		current_skill = current_skill - improvement_factor*day_changes[day]
		score_frac = rng_gen.normal(loc=current_skill, scale=0.05, size=None)
		score_frac = score_frac if score_frac < 1 else 1
		archer_list[shared.COLUMN_SCORE].append(score_frac)


df = pd.DataFrame(archer_list)
df.to_csv(shared.PATH_DATASET, index=False)
