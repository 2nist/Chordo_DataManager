import os
import pandas as pd
from clean_data import clean_data
from chord_normalizer import normalize_chords_and_deduplicate
from spotify_integration import get_and_populate_audio_features
from audio_features import extract_audio_features
from version_tracking import initialize_version_columns, track_changes

def step_1_clean_data(input_file, output_file):
    """Step 1: Clean the dataset."""
    print("Step 1: Cleaning the dataset...")
    
    # Load and initialize version tracking for new dataset
    df = pd.read_parquet(input_file)
    df = initialize_version_columns(df)
    
    # Clean the data and track changes
    cleaned_df = clean_data(df)
    cleaned_df = track_changes(df, cleaned_df)
    
    cleaned_df.to_parquet(output_file)
    print(f"Cleaned data saved to {output_file}")

def step_2_normalize_chords(input_file, output_file, chord_column='chord'):
    """Step 2: Normalize chords and remove duplicates."""
    print("Step 2: Normalizing chords and removing duplicates...")
    df = pd.read_parquet(input_file)
    
    # Normalize chords (version tracking is handled inside the function)
    normalized_df = normalize_chords_and_deduplicate(df, chord_column)
    normalized_df.to_parquet(output_file)
    print(f"Normalized data saved to {output_file}")

def step_3_spotify_integration(input_file, output_file):
    """Step 3: Integrate with Spotify to fetch additional data."""
    print("Step 3: Integrating with Spotify API...")
    df = pd.read_parquet(input_file)
    
    # Get Spotify data and track changes
    original_df = df.copy()
    enhanced_df = get_and_populate_audio_features(df, 'song_title')
    enhanced_df = track_changes(original_df, enhanced_df)
    
    enhanced_df.to_parquet(output_file)
    print(f"Spotify-enhanced data saved to {output_file}")

def step_4_extract_audio_features(input_file, output_file):
    """Step 4: Extract audio features."""
    print("Step 4: Extracting audio features...")
    df = pd.read_parquet(input_file)
    
    # Extract features and track changes
    original_df = df.copy()
    audio_features_df = extract_audio_features(df)
    audio_features_df = track_changes(original_df, audio_features_df)
    
    audio_features_df.to_parquet(output_file)
    print(f"Audio features saved to {output_file}")

def main():
    """Main function to orchestrate the data preparation pipeline."""
    base_data_path = '../data/'

    # File paths
    raw_data_file = os.path.join(base_data_path, 'chordonomicon.parquet')
    cleaned_data_file = os.path.join(base_data_path, 'cleaned_chordonomicon.parquet')
    normalized_data_file = os.path.join(base_data_path, 'normalized_chordonomicon.parquet')
    spotify_data_file = os.path.join(base_data_path, 'enhanced_chordonomicon.parquet')
    audio_features_file = os.path.join(base_data_path, 'audio_features.parquet')

    # Execute pipeline steps
    step_1_clean_data(raw_data_file, cleaned_data_file)
    step_2_normalize_chords(cleaned_data_file, normalized_data_file, chord_column='chord')
    step_3_spotify_integration(normalized_data_file, spotify_data_file)
    step_4_extract_audio_features(spotify_data_file, audio_features_file)

    print("Data preparation pipeline completed successfully.")

if __name__ == "__main__":
    main()