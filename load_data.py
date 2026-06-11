import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ============================================================
# STEP 1: Parse ratings file
# ============================================================

def parse_netflix_file(filepath):
    rows = []
    current_movie_id = None

    with open(filepath, 'r') as f:
        for line in tqdm(f, desc=f"Parsing {os.path.basename(filepath)}"):
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(',')
                if len(parts) == 3 and current_movie_id is not None:
                    try:
                        user_id = int(parts[0])
                        rating  = int(parts[1])
                        date    = parts[2]
                        rows.append([current_movie_id, user_id, rating, date])
                    except ValueError:
                        continue

    return pd.DataFrame(rows, columns=['movie_id', 'user_id', 'rating', 'date'])


# ============================================================
# STEP 2: Parse movie titles
# ============================================================

def parse_movie_titles(filepath):
    rows = []
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)  # split on first 2 commas only
            if len(parts) == 3:
                try:
                    movie_id = int(parts[0])
                    year     = parts[1].strip()
                    title    = parts[2].strip()
                    rows.append([movie_id, year, title])
                except ValueError:
                    continue
    return pd.DataFrame(rows, columns=['movie_id', 'year', 'title'])


# ============================================================
# STEP 3: Load ratings from combined_data_1.txt
# ============================================================

print("=" * 50)
print("STEP 1: Loading ratings...")
print("=" * 50)

df = parse_netflix_file('data/combined_data_1.txt')

print(f"\n✅ Ratings loaded successfully!")
print(f"   Total ratings  : {len(df):,}")
print(f"   Unique users   : {df['user_id'].nunique():,}")
print(f"   Unique movies  : {df['movie_id'].nunique():,}")
print(f"   Rating range   : {df['rating'].min()} to {df['rating'].max()}")
print(f"\nSample rows:")
print(df.head())


# ============================================================
# STEP 4: Load movie titles
# ============================================================

print("\n" + "=" * 50)
print("STEP 2: Loading movie titles...")
print("=" * 50)

movies = parse_movie_titles('data/movie_titles.csv')

print(f"\n✅ Movies loaded successfully!")
print(f"   Total movies : {len(movies):,}")
print(f"\nSample rows:")
print(movies.head(10))


# ============================================================
# STEP 5: Create working subset
# ============================================================

print("\n" + "=" * 50)
print("STEP 3: Creating working subset...")
print("=" * 50)

# Top 10,000 most active users
top_users = df['user_id'].value_counts().head(10000).index
df_small  = df[df['user_id'].isin(top_users)].copy()

print(f"\n✅ Subset created!")
print(f"   Ratings : {len(df_small):,}")
print(f"   Users   : {df_small['user_id'].nunique():,}")
print(f"   Movies  : {df_small['movie_id'].nunique():,}")


# ============================================================
# STEP 6: Save files
# ============================================================

print("\n" + "=" * 50)
print("STEP 4: Saving files...")
print("=" * 50)

df_small.to_csv('data/netflix_subset.csv', index=False)
movies.to_csv('data/movies_clean.csv', index=False)

# Verify files were saved
size1 = os.path.getsize('data/netflix_subset.csv') / (1024*1024)
size2 = os.path.getsize('data/movies_clean.csv') / (1024*1024)

print(f"\n✅ data/netflix_subset.csv saved  ({size1:.1f} MB)")
print(f"✅ data/movies_clean.csv  saved  ({size2:.1f} MB)")
print("\n🎉 ALL DONE! Ready for next step.")