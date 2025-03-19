import pandas as pd
from version_tracking import track_changes, initialize_version_columns

def clean_data(df):
    """
    Clean the dataset, handling missing values and normalizing numerical columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to clean

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame
    """
    # Store original state for version tracking
    original_df = df.copy()
    
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill for missing values
    
    # Normalize data (example: scaling numerical columns)
    numerical_cols = df.select_dtypes(include=['float64', 'int']).columns
    if len(numerical_cols) > 0:
        df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
    
    # Track changes
    df = track_changes(original_df, df)
    return df

def remove_incomplete_or_invalid_chords(df, chord_column='chord'):
    """
    Safely removes rows where chord data is incomplete or contains unexpected characters.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to clean.
    chord_column : str, default='chord'
        The name of the column containing chord data.

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame with invalid rows removed.
    """
    import re

    # Store original state for version tracking
    original_df = df.copy()

    # Define a regex pattern for valid chord notation (example: Cmaj7, Dm, G7, etc.)
    valid_chord_pattern = re.compile(r'^[A-G](#|b)?(maj|min|dim|aug|sus|add|m|M|7|9|11|13)?\d*$')

    # Check if the chord column exists
    if chord_column not in df.columns:
        raise ValueError(f"Column '{chord_column}' not found in DataFrame.")

    # Filter rows with valid chord data
    is_valid_chord = df[chord_column].apply(lambda x: bool(valid_chord_pattern.match(str(x))) if pd.notna(x) else False)
    cleaned_df = df[is_valid_chord].copy()

    # Track changes
    cleaned_df = track_changes(original_df, cleaned_df)
    return cleaned_df

def validate_and_clean_column_types(df, column_type_mapping):
    """
    Checks each column for unexpected data types and converts or removes invalid entries.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to validate and clean.
    column_type_mapping : dict
        A dictionary where keys are column names and values are the expected data types (e.g., 'int', 'float', 'str').

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame with invalid entries handled.
    """
    # Store original state for version tracking
    original_df = df.copy()

    for column, expected_type in column_type_mapping.items():
        if column in df.columns:
            if expected_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
            elif expected_type == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0).astype(float)
            elif expected_type == 'str':
                df[column] = df[column].astype(str)
            else:
                print(f"Warning: Unsupported type '{expected_type}' for column '{column}'.")
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    # Track changes
    df = track_changes(original_df, df)
    return df

def filter_common_chord_patterns(df, chord_column='progression'):
    """
    Filters the dataset to return rows containing common chord patterns like I-V-vi-IV, ii-V-I, or vi-IV-I-V.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter.
    chord_column : str, default='progression'
        The name of the column containing chord progressions.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame containing only rows with common chord patterns.
    """
    # Store original state for version tracking
    original_df = df.copy()

    # Define common chord patterns
    common_patterns = [
        'I-V-vi-IV',
        'ii-V-I',
        'vi-IV-I-V'
    ]

    # Check if the chord column exists
    if chord_column not in df.columns:
        raise ValueError(f"Column '{chord_column}' not found in DataFrame.")

    # Filter rows containing any of the common patterns
    filtered_df = df[df[chord_column].isin(common_patterns)].copy()

    # Track changes
    filtered_df = track_changes(original_df, filtered_df)
    return filtered_df

def filter_progressions_by_harmonic_role(df, chord_column='progression', role_column='harmonic_role'):
    """
    Filters the dataset to return rows where progressions match a specific harmonic role (Tonic, Dominant, Subdominant).

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter.
    chord_column : str, default='progression'
        The name of the column containing chord progressions.
    role_column : str, default='harmonic_role'
        The name of the column containing harmonic roles.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame containing only rows with specified harmonic roles.
    """
    # Define harmonic roles and their corresponding chords
    harmonic_roles = {
        'Tonic': ['I', 'vi', 'iii'],
        'Dominant': ['V', 'vii°'],
        'Subdominant': ['ii', 'IV']
    }

    # Check if the required columns exist
    if chord_column not in df.columns or role_column not in df.columns:
        raise ValueError(f"Required columns '{chord_column}' or '{role_column}' not found in DataFrame.")

    # Filter rows based on harmonic roles
    filtered_df = df[df[role_column].isin(harmonic_roles.keys())].copy()

    return filtered_df

def filter_high_tension_and_energy(df, tension_column='tension_score', energy_column='energy', tension_range=(4, 5), energy_threshold=0.8):
    """
    Filters the dataset for progressions with high tension scores (4-5) and high energy values (> 0.8).

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter.
    tension_column : str, default='tension_score'
        The name of the column containing tension scores.
    energy_column : str, default='energy'
        The name of the column containing energy values.
    tension_range : tuple, default=(4, 5)
        The range of tension scores to filter (inclusive).
    energy_threshold : float, default=0.8
        The minimum energy value to filter.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame containing only rows with high tension scores and high energy values.
    """
    # Check if the required columns exist
    if tension_column not in df.columns or energy_column not in df.columns:
        raise ValueError(f"Required columns '{tension_column}' or '{energy_column}' not found in DataFrame.")

    # Filter rows based on tension score and energy values
    filtered_df = df[(df[tension_column].between(tension_range[0], tension_range[1], inclusive="both")) &
                     (df[energy_column] > energy_threshold)].copy()

    return filtered_df

def filter_progressions_by_mode(df, chord_column='progression', mode_column='mode'):
    """
    Extracts progressions in Dorian, Phrygian, or Lydian modes by identifying specific modal chord movements.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter.
    chord_column : str, default='progression'
        The name of the column containing chord progressions.
    mode_column : str, default='mode'
        The name of the column containing mode information.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame containing only rows with progressions in Dorian, Phrygian, or Lydian modes.
    """
    # Define modes to filter
    target_modes = ['Dorian', 'Phrygian', 'Lydian']

    # Check if the required columns exist
    if chord_column not in df.columns or mode_column not in df.columns:
        raise ValueError(f"Required columns '{chord_column}' or '{mode_column}' not found in DataFrame.")

    # Filter rows based on the target modes
    filtered_df = df[df[mode_column].isin(target_modes)].copy()

    return filtered_df

def detect_shared_pivot_chords(df, progression_column='progression', pivot_column='pivot_chords'):
    """
    Detects shared pivot chords between progressions and recommends them for modulation points.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions.
    progression_column : str, default='progression'
        The name of the column containing chord progressions.
    pivot_column : str, default='pivot_chords'
        The name of the column to store detected pivot chords.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with an additional column for recommended pivot chords.
    """
    # Check if the progression column exists
    if progression_column not in df.columns:
        raise ValueError(f"Column '{progression_column}' not found in DataFrame.")

    # Function to find shared chords between progressions
    def find_pivot_chords(progression):
        try:
            chords = progression.split('-')
            return set(chords)
        except Exception as e:
            print(f"Error processing progression '{progression}': {e}")
            return set()

    # Extract pivot chords for each progression
    df[pivot_column] = df[progression_column].apply(find_pivot_chords)

    # Recommend pivot chords by finding intersections
    pivot_recommendations = []
    for i, row_i in df.iterrows():
        shared_chords = set()
        for j, row_j in df.iterrows():
            if i != j:  # Avoid comparing the same progression
                shared_chords.update(row_i[pivot_column].intersection(row_j[pivot_column]))
        pivot_recommendations.append(', '.join(shared_chords))

    # Add recommendations to the DataFrame
    df['recommended_pivot_chords'] = pivot_recommendations

    return df

if __name__ == "__main__":
    input_file = '../data/chordonomicon.parquet'
    df = pd.read_parquet(input_file)
    
    # Initialize version tracking for the dataset
    df = initialize_version_columns(df)
    
    # Clean the data
    cleaned_df = clean_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

    # Example with chord validation
    sample_data = {
        'chord': ['Cmaj7', 'Dm', 'G7', 'InvalidChord', None, 'A#dim'],
        'other_info': ['Info1', 'Info2', 'Info3', 'Info4', 'Info5', 'Info6']
    }
    df = pd.DataFrame(sample_data)
    df = initialize_version_columns(df)
    
    cleaned_df = remove_incomplete_or_invalid_chords(df)
    print("\nCleaned chord DataFrame:")
    print(cleaned_df)

    # Example DataFrame
    sample_data = {
        'int_column': [1, 2, 'invalid', None],
        'float_column': [1.1, 'invalid', 3.3, None],
        'str_column': ['valid', 123, None, 'another valid']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Define expected types
    column_type_mapping = {
        'int_column': 'int',
        'float_column': 'float',
        'str_column': 'str'
    }

    # Validate and clean the DataFrame
    cleaned_df = validate_and_clean_column_types(df, column_type_mapping)

    print("\nCleaned DataFrame:")
    print(cleaned_df)

    # Example DataFrame
    sample_data = {
        'progression': ['I-V-vi-IV', 'ii-V-I', 'vi-IV-I-V', 'I-IV-V', 'iii-vi-ii-V'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Filter for common chord patterns
    filtered_df = filter_common_chord_patterns(df)

    print("\nFiltered DataFrame:")
    print(filtered_df)

    # Example DataFrame
    sample_data = {
        'progression': ['I-V-vi-IV', 'ii-V-I', 'vi-IV-I-V', 'I-IV-V', 'iii-vi-ii-V'],
        'harmonic_role': ['Tonic', 'Dominant', 'Subdominant', 'Tonic', 'Dominant'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Filter for harmonic roles
    filtered_df = filter_progressions_by_harmonic_role(df)

    print("\nFiltered DataFrame:")
    print(filtered_df)

    # Example DataFrame
    sample_data = {
        'progression': ['I-V-vi-IV', 'ii-V-I', 'vi-IV-I-V', 'I-IV-V', 'iii-vi-ii-V'],
        'tension_score': [4.5, 3.0, 4.8, 2.5, 5.0],
        'energy': [0.9, 0.7, 0.85, 0.6, 0.95],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Filter for high tension and energy
    filtered_df = filter_high_tension_and_energy(df)

    print("\nFiltered DataFrame:")
    print(filtered_df)

    # Example DataFrame
    sample_data = {
        'progression': ['i-iv-VII-III', 'ii-vi-iii-V', 'I-II-iii-IV', 'i-ii-III-VI'],
        'mode': ['Dorian', 'Phrygian', 'Lydian', 'Aeolian'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Filter for progressions in Dorian, Phrygian, or Lydian modes
    filtered_df = filter_progressions_by_mode(df)

    print("\nFiltered DataFrame:")
    print(filtered_df)

    # Example DataFrame
    sample_data = {
        'progression': ['I-IV-V-I', 'ii-V-I', 'vi-IV-I-V', 'I-V-vi-IV'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Detect shared pivot chords
    df_with_pivots = detect_shared_pivot_chords(df)

    print("\nDataFrame with Recommended Pivot Chords:")
    print(df_with_pivots)