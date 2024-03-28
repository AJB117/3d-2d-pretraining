import pandas as pd
import os

# gather all the files in the current directory ending with .csv
files = [f for f in os.listdir(".") if f.endswith(".csv")]
# read each file into a pandas dataframe
print(files)
dfs = [pd.read_csv(f) for f in files]
# concatenate all the dataframes into one
df = pd.concat(dfs, ignore_index=True)
# write the final dataframe to a new csv file
df.to_csv("configs.csv")
