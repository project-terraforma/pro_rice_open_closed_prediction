from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"

RANDOM_STATE = 42

df = pd.read_parquet(DATA_DIR / "project_c_samples.parquet")

# Drop any columns that are completely null
null_cols = df.columns[df.isna().all()].tolist()
if null_cols:
    print(f"Dropping fully-null columns: {null_cols}")
    df = df.drop(columns=null_cols)
else:
    print("No fully-null columns to drop.")

df.head() 
df.columns 
df.info()

# Quick label sanity check
print("Label distribution (counts):")
print(df["open"].value_counts())
print("\nLabel distribution (proportions):")
print(df["open"].value_counts(normalize=True))

# 70 / 30
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["open"],
    random_state=RANDOM_STATE,
)

# temp is 30%; split into 20% val and 10% test => test is 1/3 of temp
val_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,
    stratify=temp_df["open"],
    random_state=RANDOM_STATE,
)

def show_split_stats(name, split_df):
    print(f"\n{name} size: {len(split_df)}")
    print(split_df["open"].value_counts())
    print(split_df["open"].value_counts(normalize=True))

show_split_stats("Train", train_df)
show_split_stats("Validation", val_df)
show_split_stats("Test", test_df)

# Save splits
train_df.to_parquet(DATA_DIR / "train_split.parquet", index=False)
val_df.to_parquet(DATA_DIR / "val_split.parquet", index=False)
test_df.to_parquet(DATA_DIR / "test_split.parquet", index=False)

# Save IDs too (useful for teammates + reproducibility)
train_df[["id"]].to_csv(DATA_DIR / "train_ids.csv", index=False)
val_df[["id"]].to_csv(DATA_DIR / "val_ids.csv", index=False)
test_df[["id"]].to_csv(DATA_DIR / "test_ids.csv", index=False)

print("\nSaved: train/val/test parquet + id CSVs")


'''
OUTPUT:

Dropping fully-null columns: ['emails']
<class 'pandas.DataFrame'>
RangeIndex: 3425 entries, 0 to 3424
Data columns (total 15 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   id          3425 non-null   str    
 1   geometry    3425 non-null   object 
 2   bbox        3425 non-null   object 
 3   type        3425 non-null   str    
 4   version     3425 non-null   int32  
 5   sources     3425 non-null   object 
 6   names       3425 non-null   object 
 7   categories  3425 non-null   object 
 8   confidence  3425 non-null   float64
 9   websites    3056 non-null   object 
 10  socials     2815 non-null   object 
 11  phones      3301 non-null   object 
 12  brand       575 non-null    object 
 13  addresses   3425 non-null   object 
 14  open        3425 non-null   int64  
dtypes: float64(1), int32(1), int64(1), object(10), str(2)
memory usage: 511.9+ KB
Label distribution (counts):
open
1    3112
0     313
Name: count, dtype: int64

Label distribution (proportions):
open
1    0.908613
0    0.091387
Name: proportion, dtype: float64

Train size: 2397
open
1    2178
0     219
Name: count, dtype: int64
open
1    0.908636
0    0.091364
Name: proportion, dtype: float64

Validation size: 685
open
1    622
0     63
Name: count, dtype: int64
open
1    0.908029
0    0.091971
Name: proportion, dtype: float64

Test size: 343
open
1    312
0     31
Name: count, dtype: int64
open
1    0.909621
0    0.090379
Name: proportion, dtype: float64

Saved: train/val/test parquet + id CSVs
'''