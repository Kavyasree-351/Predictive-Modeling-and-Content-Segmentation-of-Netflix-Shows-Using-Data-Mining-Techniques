import pandas as pd
import numpy as np


FILE_PATH = r'C:\Users\Lenovo\Desktop\DM PROJECT\netflix_movies_detailed_up_to_2025.csv'
OUT_PATH = r'C:\Users\Lenovo\Desktop\DM PROJECT\netflix_preprocessed_simplified.csv'

COLUMNS = [
    'show_id', 'type', 'title', 'director', 'cast', 'country',
    'date_added', 'release_year', 'rating', 'duration',
    'genres', 'language', 'description', 'popularity', 'vote_count',
    'vote_average', 'budget', 'revenue'
]

def load_and_inspect_data(file_path):
    print("--- 1. Loading and Initial Inspection ---")
    df = pd.read_csv(file_path, header=None)
    df.columns = COLUMNS
    print(f"Shape of the dataset: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
    print("\nData Types:")
    print(df.info())
    print("\nMissing Values Count:")
    print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
    return df

def initial_cleaning_and_conversion(df):
    print("\n--- 2. Initial Cleaning and Type Conversion ---")
    df['date_added'] = df['date_added'].astype(str)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['release_year'] = pd.to_numeric(df['release_year'].astype(str), errors='coerce').astype('Int64')
    numerical_cols = ['rating', 'popularity', 'vote_count', 'vote_average', 'budget', 'revenue']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df['duration'] = df['duration'].astype(str)
    print("\nData Types after conversion:")
    print(df.dtypes.to_markdown(numalign="left", stralign="left"))
    return df

def handle_missing_values_and_feature_engineering(df):
    print("\n--- 3. Handling Missing Values and Feature Engineering ---")
    for col in ['title', 'director', 'cast', 'country', 'genres', 'language', 'description', 'duration']:
        df[col] = df[col].fillna('Unknown')
    df['description'] = df['description'].astype(str)
    df['budget'] = df['budget'].fillna(0)
    df['revenue'] = df['revenue'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].median())
    df['rating'] = df['rating'].fillna(df['rating'].median())
    df['primary_genre'] = df['genres'].apply(lambda x: x.split(',')[0].strip() if x != 'Unknown' else 'Unknown')
    df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0].strip() if x != 'Unknown' else 'Unknown')
    df['has_financial_data'] = np.where((df['budget'] > 0) | (df['revenue'] > 0), 1, 0)
    print("\nMissing Values Count after imputation:")
    print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
    print("\nValue Counts for new 'primary_genre' feature (Top 10):")
    print(df['primary_genre'].value_counts().head(10).to_markdown(numalign="left", stralign="left"))
    return df

def one_hot_and_keep_true(df):
    # One-hot encode primary_country and language columns
    df = pd.get_dummies(df, columns=['primary_country', 'language'], prefix=['primary_country', 'language'])
    # Collect all one-hot columns for country and language
    country_cols = [col for col in df.columns if col.startswith('primary_country_')]
    language_cols = [col for col in df.columns if col.startswith('language_')]
    # Add new columns with only the TRUE/active item name
    def get_flagged(row, cols, tag):
        for col in cols:
            if row[col] == 1:
                return col.replace(f"{tag}_", "")
        return "Unknown"
    df['primary_country_flagged'] = df.apply(lambda row: get_flagged(row, country_cols, "primary_country"), axis=1)
    df['language_flagged'] = df.apply(lambda row: get_flagged(row, language_cols, "language"), axis=1)
    # Drop all the one-hot columns for country/language
    df.drop(country_cols + language_cols, axis=1, inplace=True)
    return df

def main():
    try:
        df = load_and_inspect_data(FILE_PATH)
        df = initial_cleaning_and_conversion(df)
        df = handle_missing_values_and_feature_engineering(df)
        df = one_hot_and_keep_true(df)
        df.to_csv(OUT_PATH, index=False)
        print("\n--- 4. Preprocessing Complete ---")
        print(f"Preprocessed data saved to '{OUT_PATH}'")
        print("Final column list:")
        print(df.columns.to_list())
    except FileNotFoundError:
        print(f"Error: The file {FILE_PATH} was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
