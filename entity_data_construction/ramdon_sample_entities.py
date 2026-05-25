import pandas as pd
import random
from pathlib import Path

# set random seed
random.seed(42)

original_csv = '/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/user_unique_entities.csv'

# read csv
df = pd.read_csv(original_csv)

# randomly sample 500 rows
sampled_df = df.sample(n=500, random_state=42)

# build output path (same directory)
output_path = Path(original_csv).parent / 'user_unique_entities_sampled_500.csv'

# save to csv
sampled_df.to_csv(output_path, index=False)
