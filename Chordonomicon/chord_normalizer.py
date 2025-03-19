import pandas as pd
from music21 import harmony
from chordonomicon_prep.scripts.version_tracking import initialize_version_columns, track_changes

def normalize_chords_and_deduplicate(df, chord_column='chord'):
    """
    Normalize chord notation using music21 and remove duplicates from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord data
    chord_column : str, default='chord'
        The name of the column containing chord notations
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with normalized chord notation and duplicates removed
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Function to normalize a chord using music21
    def normalize_chord(chord_str):
        try:
            # Parse the chord with music21
            chord_obj = harmony.ChordSymbol(chord_str)
            
            # Get the normalized common name
            normalized_chord = chord_obj.figure
            return normalized_chord
        except Exception as e:
            # Return original if music21 cannot parse it
            print(f"Warning: Could not normalize chord '{chord_str}': {e}")
            return chord_str
    
    # Apply the normalization function to the chord column
    df_copy[chord_column] = df_copy[chord_column].apply(normalize_chord)
    
    # Remove duplicates
    df_deduped = df_copy.drop_duplicates()
    
    # Track version changes
    df_deduped = track_changes(df, df_deduped)
    
    return df_deduped

def merge_dataframes_with_priority(dataframes, join_column='progression_id'):
    """
    Merges multiple DataFrames based on a common join column and keeps only non-null values.
    When multiple DataFrames have values for the same field, values from the earlier DataFrame
    in the list take priority.
    
    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        A list of DataFrames to merge
    join_column : str, default='progression_id'
        The column to join the DataFrames on
        
    Returns:
    --------
    pandas.DataFrame
        A merged DataFrame with non-null values from each dataset
    """
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        # Initialize version tracking for single DataFrame
        return initialize_version_columns(dataframes[0].copy())
    
    # Start with the first DataFrame
    result_df = dataframes[0].copy()
    result_df = initialize_version_columns(result_df)
    
    # Merge with each subsequent DataFrame
    for df in dataframes[1:]:
        # Initialize version tracking for incoming DataFrame if needed
        df = initialize_version_columns(df)
        
        # Check if join_column exists in both DataFrames
        if join_column not in result_df.columns or join_column not in df.columns:
            raise ValueError(f"Join column '{join_column}' not found in one of the DataFrames")
        
        # Store original state for version tracking
        original_df = result_df.copy()
        
        # Perform an outer join to keep all rows from both DataFrames
        merged = result_df.merge(df, on=join_column, how='outer', suffixes=('', '_new'))
        
        # Handle version columns specially
        version_cols = ['version', 'created_at', 'last_updated']
        data_cols = [col for col in df.columns if col not in version_cols and col != join_column]
        
        for col in data_cols:
            # Column name after merge will have '_new' suffix
            new_col = f"{col}_new"
            
            if col in result_df.columns and new_col in merged.columns:
                # Fill null values in the original column with values from the new column
                merged[col] = merged[col].fillna(merged[new_col])
                
                # Drop the duplicate column
                merged = merged.drop(new_col, axis=1)
        
        # Update version information for changed rows
        result_df = track_changes(original_df, merged)
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame with chord notations
    data = {
        'chord': ['Cmaj7', 'C Major 7', 'Dm7', 'D-7', 'G7', 'G dom7'],
        'other_info': ['Example 1', 'Example 2', 'Example 3', 'Example 4', 'Example 5', 'Example 6']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Normalize chords and remove duplicates
    normalized_df = normalize_chords_and_deduplicate(df)
    print("\nNormalized DataFrame with duplicates removed:")
    print(normalized_df)
    
    # Example of merging DataFrames with priority
    df1 = pd.DataFrame({
        'progression_id': [1, 2, 3, 4],
        'artist': ['Artist A', None, 'Artist C', 'Artist D'],
        'tempo': [120, None, 140, None]
    })
    
    df2 = pd.DataFrame({
        'progression_id': [1, 2, 3, 5],
        'artist': [None, 'Artist B', 'Different Artist', 'Artist E'],
        'energy': [0.8, 0.7, 0.9, 0.6]
    })
    
    df3 = pd.DataFrame({
        'progression_id': [1, 3, 5, 6],
        'tempo': [None, 150, 90, 110],
        'key': ['C', 'G', 'D', 'A']
    })
    
    print("\nMerging multiple DataFrames example:")
    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    print("\nDataFrame 3:")
    print(df3)
    
    merged_df = merge_dataframes_with_priority([df1, df2, df3])
    print("\nMerged DataFrame (non-null values preserved):")
    print(merged_df)
