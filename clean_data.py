import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Load saved files
# ============================================================

print("Loading saved data...")
df     = pd.read_csv('data/netflix_subset.csv')
movies = pd.read_csv('data/movies_clean.csv')

print(f"Ratings shape : {df.shape}")
print(f"Movies shape  : {movies.shape}")


# ============================================================
# STEP 2: Check missing values
# ============================================================

print("\n--- Missing Values ---")
print("Ratings:\n", df.isnull().sum())
print("\nMovies:\n", movies.isnull().sum())


# ============================================================
# STEP 3: Check and remove duplicates
# ============================================================

print("\n--- Duplicate Check ---")
dupes = df.duplicated(subset=['user_id', 'movie_id']).sum()
print(f"Duplicate user-movie pairs : {dupes:,}")

if dupes > 0:
    df = df.sort_values('date').drop_duplicates(
        subset=['user_id', 'movie_id'], keep='last'
    )
    print(f"After dropping dupes : {len(df):,} ratings")
else:
    print("No duplicates ✅")


# ============================================================
# STEP 4: Fix data types
# ============================================================

print("\n--- Fixing Data Types ---")

df['date']     = pd.to_datetime(df['date'])
df['rating']   = df['rating'].astype(int)
df['user_id']  = df['user_id'].astype(int)
df['movie_id'] = df['movie_id'].astype(int)

movies['year'] = movies['year'].replace('NULL', np.nan)
movies['year'] = pd.to_numeric(movies['year'], errors='coerce')

print("✅ Data types fixed")


# ============================================================
# STEP 5: Remove low-activity users (rated fewer than 5 movies)
# ============================================================

print("\n--- Filtering Low-Activity Users ---")
before = len(df)
user_counts  = df['user_id'].value_counts()
active_users = user_counts[user_counts >= 5].index
df = df[df['user_id'].isin(active_users)]
print(f"Removed {before - len(df):,} ratings from low-activity users")
print(f"Remaining ratings : {len(df):,}")


# ============================================================
# STEP 6: Remove rarely rated movies (fewer than 10 ratings)
# ============================================================

print("\n--- Filtering Rarely Rated Movies ---")
before = len(df)
movie_counts    = df['movie_id'].value_counts()
popular_movies  = movie_counts[movie_counts >= 10].index
df = df[df['movie_id'].isin(popular_movies)]
print(f"Removed {before - len(df):,} ratings from rare movies")
print(f"Remaining ratings : {len(df):,}")


# ============================================================
# STEP 7: Add derived columns
# ============================================================

print("\n--- Adding Derived Columns ---")
df['rating_year']  = df['date'].dt.year
df['rating_month'] = df['date'].dt.month
print("✅ Added rating_year and rating_month")


# ============================================================
# STEP 8: Calculate sparsity
# ============================================================

n_users  = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()
n_ratings = len(df)
sparsity = 100 * (1 - n_ratings / (n_users * n_movies))


# ============================================================
# STEP 9: Final summary
# ============================================================

print("\n" + "=" * 50)
print("CLEAN DATA SUMMARY")
print("=" * 50)
print(f"Total ratings  : {n_ratings:,}")
print(f"Unique users   : {n_users:,}")
print(f"Unique movies  : {n_movies:,}")
print(f"Rating range   : {df['rating'].min()} to {df['rating'].max()}")
print(f"Date range     : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Sparsity       : {sparsity:.2f}%")
print(f"\nRating distribution:")
print(df['rating'].value_counts().sort_index())


# ============================================================
# STEP 10: Save clean data
# ============================================================

df.to_csv('data/netflix_clean.csv', index=False)
movies.to_csv('data/movies_clean.csv', index=False)

print("\n✅ Saved data/netflix_clean.csv")
print("🎉 Data is clean and ready for EDA!")