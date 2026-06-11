import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

# ============================================================
# Setup
# ============================================================

def save_fig(fname, dpi=150):
    """Helper to save figures into the chosen plots directory (plots_dir)."""
    global plots_dir
    path = os.path.join(plots_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="EDA for Netflix recsys (generate plots).")
    parser.add_argument("--ratings", "-r", default="data/netflix_clean.csv", help="Ratings CSV path")
    parser.add_argument("--movies", "-m", default="data/movies_clean.csv", help="Movies CSV path")
    parser.add_argument("--plots-dir", "-p", default="plots", help="Directory to save plots")
    args = parser.parse_args()

    global plots_dir
    plots_dir = args.plots_dir
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    print("Loading clean data...")
    if not os.path.exists(args.ratings):
        print(f"Ratings file not found: {args.ratings}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.movies):
        print(f"Movies file not found: {args.movies}", file=sys.stderr)
        sys.exit(1)

    df     = pd.read_csv(args.ratings)
    movies = pd.read_csv(args.movies)
    df['date'] = pd.to_datetime(df['date'])
    # Ensure rating_year exists (compute from date if missing)
    if 'rating_year' not in df.columns:
        df['rating_year'] = df['date'].dt.year
    print(f"✅ Loaded {len(df):,} ratings")

    # ============================================================
    # PLOT 1: Rating Distribution
    # ============================================================

    plt.figure(figsize=(8,5))
    counts = df['rating'].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette='Blues_d')
    plt.title('Rating Distribution', fontsize=14)
    plt.xlabel('Rating (1-5)')
    plt.ylabel('Number of Ratings')
    for i, v in enumerate(counts.values):
        plt.text(i, v + 10000, f'{v:,}', ha='center', fontsize=9)
    save_fig('01_rating_distribution.png')

    # ============================================================
    # PLOT 2: Ratings Per User (User Activity)
    # ============================================================

    user_activity = df['user_id'].value_counts()

    plt.figure(figsize=(8,5))
    plt.hist(user_activity.values, bins=50, color='steelblue', edgecolor='white')
    plt.title('User Activity Distribution\n(How many movies each user rated)', fontsize=13)
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.axvline(user_activity.median(), color='red', linestyle='--', label=f'Median: {user_activity.median():.0f}')
    plt.legend()
    save_fig('02_user_activity.png')

    # ============================================================
    # PLOT 3: Ratings Per Movie (Movie Popularity)
    # ============================================================

    movie_popularity = df['movie_id'].value_counts()

    plt.figure(figsize=(8,5))
    plt.hist(movie_popularity.values, bins=50, color='coral', edgecolor='white')
    plt.title('Movie Popularity Distribution\n(How many ratings each movie received)', fontsize=13)
    plt.xlabel('Number of Ratings per Movie')
    plt.ylabel('Number of Movies')
    plt.axvline(movie_popularity.median(), color='blue', linestyle='--', label=f'Median: {movie_popularity.median():.0f}')
    plt.legend()
    save_fig('03_movie_popularity.png')

    # ============================================================
    # PLOT 4: Top 20 Most Rated Movies
    # ============================================================

    top20_movies = (df.groupby('movie_id')['rating']
                      .count()
                      .reset_index()
                      .rename(columns={'rating':'num_ratings'})
                      .merge(movies[['movie_id','title']], on='movie_id')
                      .sort_values('num_ratings', ascending=False)
                      .head(20))
    plt.figure(figsize=(10,7))
    sns.barplot(data=top20_movies, x='num_ratings', y='title', palette='viridis')
    plt.title('Top 20 Most Rated Movies', fontsize=14)
    plt.xlabel('Number of Ratings')
    plt.ylabel('')
    save_fig('04_top20_movies.png')

    # ============================================================
    # PLOT 5: Average Rating per Movie (Top 20)
    # ============================================================

    top20_avg = (df.groupby('movie_id')['rating']
                   .agg(['mean','count'])
                   .reset_index()
                   .query('count >= 100')
                   .merge(movies[['movie_id','title']], on='movie_id')
                   .sort_values('mean', ascending=False)
                   .head(20))
    plt.figure(figsize=(10,7))
    sns.barplot(data=top20_avg, x='mean', y='title', palette='RdYlGn')
    plt.title('Top 20 Highest Rated Movies\n(minimum 100 ratings)', fontsize=13)
    plt.xlabel('Average Rating')
    plt.ylabel('')
    plt.xlim(3.5, 5.0)
    save_fig('05_top20_avg_rating.png')

    # ============================================================
    # PLOT 6: Ratings Over Time
    # ============================================================

    ratings_over_time = df.groupby('rating_year')['rating'].count()

    plt.figure(figsize=(10,5))
    plt.plot(ratings_over_time.index, ratings_over_time.values, marker='o', color='steelblue', linewidth=2)
    plt.fill_between(ratings_over_time.index, ratings_over_time.values, alpha=0.2, color='steelblue')
    plt.title('Number of Ratings Over Time (by Year)', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Number of Ratings')
    save_fig('06_ratings_over_time.png')

    # ============================================================
    # PLOT 7: Average Rating Over Time
    # ============================================================

    avg_over_time = df.groupby('rating_year')['rating'].mean()

    plt.figure(figsize=(10,5))
    plt.plot(avg_over_time.index, avg_over_time.values, marker='o', color='coral', linewidth=2)
    plt.title('Average Rating Over Time (by Year)', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.ylim(1, 5)
    save_fig('07_avg_rating_over_time.png')

    # ============================================================
    # PLOT 8: Sparsity Visualization
    # ============================================================

    sample_users  = df['user_id'].value_counts().head(50).index
    sample_movies = df['movie_id'].value_counts().head(50).index
    sample_df     = df[df['user_id'].isin(sample_users) & df['movie_id'].isin(sample_movies)]
    matrix_sample = sample_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix_sample.notna(), cbar=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Data Sparsity (Blue = Rating Exists)\nTop 50 Users × Top 50 Movies', fontsize=13)
    plt.xlabel('Movies')
    plt.ylabel('Users')
    save_fig('08_sparsity_heatmap.png')

    # ============================================================
    # Print Key Statistics
    # ============================================================

    n_users   = df['user_id'].nunique()
    n_movies  = df['movie_id'].nunique()
    n_ratings = len(df)
    sparsity  = 100 * (1 - n_ratings / (n_users * n_movies))

    print("\n" + "="*50)
    print("EDA KEY STATISTICS")
    print("="*50)
    print(f"Total Ratings       : {n_ratings:,}")
    print(f"Unique Users        : {n_users:,}")
    print(f"Unique Movies       : {n_movies:,}")
    print(f"Sparsity            : {sparsity:.2f}%")
    print(f"Avg Ratings/User    : {user_activity.mean():.1f}")
    print(f"Median Ratings/User : {user_activity.median():.0f}")
    print(f"Avg Ratings/Movie   : {movie_popularity.mean():.1f}")
    print(f"Most Common Rating  : {df['rating'].mode()[0]}")
    print(f"Overall Avg Rating  : {df['rating'].mean():.3f}")
    print(f"\n✅ All 8 plots saved in {plots_dir} folder")
    print("🎉 EDA Complete! Ready for modelling.")

if __name__ == "__main__":
    main()