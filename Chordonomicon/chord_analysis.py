import music21
from music21 import harmony, roman, key, chord
from typing import Tuple, List, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np

def analyze_cadence(progression: str, key_signature: str = 'C') -> dict:
    """
    Analyzes a chord progression, identifies its cadence type, and labels the cadence strength.
    
    Parameters:
    -----------
    progression : str
        A string representation of the chord progression (e.g., "I-IV-V-I", "ii-V-I").
        Use standard Roman numeral notation.
    key_signature : str, default='C'
        The key signature for the progression (e.g., 'C', 'G', 'F#m').
        
    Returns:
    --------
    dict
        A dictionary containing cadence information:
        - 'cadence_type': The type of cadence (Perfect Authentic, Half, Deceptive, etc.)
        - 'cadence_strength': The strength rating ('Strong', 'Moderate', 'Weak')
        - 'final_chords': The final two chords that form the cadence
        - 'analysis': Additional analysis information
    """
    # Parse the progression into a list of chords
    chord_list = progression.split('-')
    
    # Need at least two chords to identify a cadence
    if len(chord_list) < 2:
        return {
            'cadence_type': 'Insufficient chords',
            'cadence_strength': 'Unknown',
            'final_chords': chord_list,
            'analysis': 'At least two chords are required to identify a cadence'
        }
    
    # Extract the last two chords in the progression
    penultimate_chord = chord_list[-2]
    final_chord = chord_list[-1]
    
    # Determine the key
    k = None
    if key_signature.endswith('m'):
        # Minor key
        k = key.Key(key_signature[:-1], 'minor')
    else:
        # Major key
        k = key.Key(key_signature)
    
    # Create Roman numeral objects for the last two chords
    try:
        # Convert roman numerals to actual chord objects
        penultimate = roman.RomanNumeral(penultimate_chord, k)
        final = roman.RomanNumeral(final_chord, k)
    except Exception as e:
        return {
            'cadence_type': 'Error',
            'cadence_strength': 'Unknown',
            'final_chords': [penultimate_chord, final_chord],
            'analysis': f'Error parsing chord: {str(e)}'
        }
    
    # Analyze the cadence type
    cadence_type = 'Other'
    cadence_strength = 'Weak'
    
    # Perfect Authentic Cadence (PAC): V-I with both chords in root position
    # and soprano ends on tonic
    if (penultimate.figure == 'V' and final.figure == 'I' and 
        penultimate.inversion() == 0 and final.inversion() == 0):
        cadence_type = 'Perfect Authentic Cadence (PAC)'
        cadence_strength = 'Strong'
    
    # Imperfect Authentic Cadence (IAC): V-I but either chord is inverted
    # or soprano doesn't land on tonic
    elif (penultimate.figure in ['V', 'V7'] and final.figure == 'I' and 
          (penultimate.inversion() != 0 or final.inversion() != 0)):
        cadence_type = 'Imperfect Authentic Cadence (IAC)'
        cadence_strength = 'Moderate'
    
    # Half Cadence (HC): any chord to V
    elif final.figure == 'V' and final.inversion() == 0:
        cadence_type = 'Half Cadence (HC)'
        cadence_strength = 'Moderate'
    
    # Deceptive Cadence: V-vi
    elif penultimate.figure in ['V', 'V7'] and final.figure == 'vi':
        cadence_type = 'Deceptive Cadence'
        cadence_strength = 'Weak'
    
    # Plagal Cadence: IV-I
    elif penultimate.figure == 'IV' and final.figure == 'I':
        cadence_type = 'Plagal Cadence'
        cadence_strength = 'Moderate'
    
    # Phrygian Half Cadence: iv6-V in minor
    elif (k.mode == 'minor' and penultimate.figure == 'iv6' and 
          final.figure == 'V'):
        cadence_type = 'Phrygian Half Cadence'
        cadence_strength = 'Moderate'
    
    return {
        'cadence_type': cadence_type,
        'cadence_strength': cadence_strength,
        'final_chords': [penultimate_chord, final_chord],
        'analysis': f'{penultimate_chord}-{final_chord} forms a {cadence_type}'
    }

def analyze_progressions_in_dataframe(df, progression_column='progression', key_column='key_signature'):
    """
    Analyzes cadences for all progressions in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions.
    progression_column : str, default='progression'
        The name of the column containing chord progressions.
    key_column : str, default='key_signature'
        The name of the column containing key signatures.
        
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with added cadence analysis columns.
    """
    import pandas as pd
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add new columns if they don't exist
    if 'cadence_type' not in result_df.columns:
        result_df['cadence_type'] = None
    if 'cadence_strength' not in result_df.columns:
        result_df['cadence_strength'] = None
    
    # Process each row
    for idx, row in result_df.iterrows():
        # Get the progression and key signature
        progression = row[progression_column]
        key_sig = 'C'  # Default key
        
        if key_column in row and row[key_column]:
            key_sig = row[key_column]
        
        # Analyze the cadence
        analysis = analyze_cadence(progression, key_sig)
        
        # Update the DataFrame
        result_df.at[idx, 'cadence_type'] = analysis['cadence_type']
        result_df.at[idx, 'cadence_strength'] = analysis['cadence_strength']
    
    return result_df

def analyze_chord_inversions(chord_progression: str, key_signature: str = 'C') -> dict:
    """
    Detects chord inversions in a progression and maps them to tension scores based on inversion position.
    
    Parameters:
    -----------
    chord_progression : str
        A string representation of the chord progression (e.g., "I-IV-V-I", "ii-V-I").
        Use standard Roman numeral notation.
    key_signature : str, default='C'
        The key signature for the progression (e.g., 'C', 'G', 'F#m').
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'chords': List of dictionaries with chord information (Roman numeral, inversion, tension score)
        - 'average_tension': The average tension score across the progression
        - 'highest_tension': The highest tension score in the progression
        - 'progression_profile': Description of the tension arc
    """
    # Parse the progression into a list of chords
    chord_list = chord_progression.split('-')
    
    # Determine the key
    k = None
    if key_signature.endswith('m'):
        # Minor key
        k = key.Key(key_signature[:-1], 'minor')
    else:
        # Major key
        k = key.Key(key_signature)
    
    # Define tension mappings for different chord types and inversions
    # Format: (chord type, inversion) -> tension_score
    tension_map = {
        # Major triads
        ('M', 0): 1.0,  # Root position - most stable
        ('M', 1): 1.5,  # First inversion - adds slight tension
        ('M', 2): 2.0,  # Second inversion - more tension
        
        # Minor triads
        ('m', 0): 1.2,  # Root position - slightly more tense than major
        ('m', 1): 1.7,  # First inversion
        ('m', 2): 2.2,  # Second inversion
        
        # Dominant seventh
        ('7', 0): 2.5,  # Root position - creates tension
        ('7', 1): 3.0,  # First inversion - adds more tension
        ('7', 2): 3.5,  # Second inversion - significant tension
        ('7', 3): 4.0,  # Third inversion - high tension
        
        # Major seventh
        ('M7', 0): 1.8,  # Root position - stable but with color
        ('M7', 1): 2.3,  # First inversion
        ('M7', 2): 2.8,  # Second inversion
        ('M7', 3): 3.3,  # Third inversion
        
        # Minor seventh
        ('m7', 0): 2.0,  # Root position
        ('m7', 1): 2.5,  # First inversion
        ('m7', 2): 3.0,  # Second inversion
        ('m7', 3): 3.5,  # Third inversion
        
        # Diminished triad
        ('o', 0): 3.2,  # Root position - tense
        ('o', 1): 3.7,  # First inversion
        ('o', 2): 4.2,  # Second inversion
        
        # Diminished seventh
        ('o7', 0): 3.8,  # Root position - very tense
        ('o7', 1): 4.0,  # First inversion
        ('o7', 2): 4.2,  # Second inversion
        ('o7', 3): 4.4,  # Third inversion
        
        # Half-diminished seventh
        ('ø7', 0): 3.5,  # Root position
        ('ø7', 1): 3.7,  # First inversion
        ('ø7', 2): 3.9,  # Second inversion
        ('ø7', 3): 4.1,  # Third inversion
        
        # Augmented triad
        ('+', 0): 3.5,  # Root position - quite tense
        ('+', 1): 3.5,  # First inversion (same as root due to symmetry)
        ('+', 2): 3.5,  # Second inversion (same as root due to symmetry)
    }
    
    # Default tension score for unknown chord types
    default_tension = 2.5
    
    # Extract chord information and calculate tension
    chord_details = []
    tension_scores = []
    
    for roman_numeral in chord_list:
        try:
            # Parse the chord symbol to extract base figure and inversion
            base_figure = roman_numeral
            inversion = 0
            
            # Extract inversion information if present
            # Example: V6 (first inversion), V64 (second inversion), V7 (root), V65 (first inv of 7th), etc.
            if '43' in roman_numeral:  # Third inversion of seventh chord
                base_figure = roman_numeral.replace('43', '')
                inversion = 3
            elif '42' in roman_numeral:  # Third inversion of seventh chord (alternate notation)
                base_figure = roman_numeral.replace('42', '')
                inversion = 3
            elif '65' in roman_numeral:  # First inversion of seventh chord
                base_figure = roman_numeral.replace('65', '')
                inversion = 1
            elif '64' in roman_numeral:  # Second inversion of triad
                base_figure = roman_numeral.replace('64', '')
                inversion = 2
            elif '6' in roman_numeral and not '65' in roman_numeral:  # First inversion of triad
                base_figure = roman_numeral.replace('6', '')
                inversion = 1
            
            # Create Roman numeral object for scale degree
            rn = roman.RomanNumeral(base_figure, k)
            
            # Determine chord type based on figure
            chord_type = 'M'  # Default to major
            if rn.quality == 'minor':
                chord_type = 'm'
            elif rn.quality == 'diminished':
                chord_type = 'o'
            elif rn.quality == 'augmented':
                chord_type = '+'
                
            # Check for seventh chords
            if '7' in roman_numeral:
                if chord_type == 'M':
                    if 'maj7' in roman_numeral.lower() or 'M7' in roman_numeral:
                        chord_type = 'M7'
                    else:
                        chord_type = '7'  # Dominant seventh
                elif chord_type == 'm':
                    chord_type = 'm7'
                elif chord_type == 'o':
                    chord_type = 'o7'
                    
            # Half-diminished handling
            if 'ø' in roman_numeral or 'ø7' in roman_numeral:
                chord_type = 'ø7'
            
            # Look up tension score
            tension_score = tension_map.get((chord_type, inversion), default_tension)
            
            # Function degree can affect tension (e.g., V is more tense than I)
            degree = rn.scaleDegree
            
            # Adjust tension based on scale degree
            degree_tension = {
                1: 0.0,    # Tonic (I/i) - most stable
                3: 0.2,    # Mediant (iii/III) - relatively stable
                6: 0.3,    # Submediant (vi/VI) - relatively stable
                4: 0.4,    # Subdominant (IV/iv) - mild tension
                2: 0.5,    # Supertonic (ii/II) - mild tension
                5: 0.7,    # Dominant (V/v) - creates tension
                7: 0.9     # Leading tone (vii/VII) - high tension
            }
            
            # Add degree-based tension adjustment
            adjusted_tension = tension_score + degree_tension.get(degree, 0.0)
            
            # Cap tension at 5.0
            adjusted_tension = min(5.0, adjusted_tension)
            
            # Get inversion name
            inversion_names = ['root position', '1st inversion', '2nd inversion', '3rd inversion']
            inversion_name = inversion_names[inversion] if inversion < len(inversion_names) else f"{inversion}th inversion"
            
            # Store chord info
            chord_details.append({
                'roman_numeral': roman_numeral,
                'chord_type': chord_type,
                'inversion': inversion,
                'inversion_name': inversion_name,
                'scale_degree': degree,
                'tension_score': round(adjusted_tension, 1)
            })
            
            tension_scores.append(adjusted_tension)
            
        except Exception as e:
            # Handle parsing errors
            chord_details.append({
                'roman_numeral': roman_numeral,
                'error': str(e),
                'tension_score': default_tension
            })
            tension_scores.append(default_tension)
    
    # Calculate average tension
    avg_tension = sum(tension_scores) / len(tension_scores) if tension_scores else 0
    max_tension = max(tension_scores) if tension_scores else 0
    
    # Analyze tension profile
    tension_profile = "Unknown"
    if len(tension_scores) >= 3:
        first_third = tension_scores[0:len(tension_scores)//3]
        last_third = tension_scores[-(len(tension_scores)//3):]
        
        first_avg = sum(first_third) / len(first_third)
        last_avg = sum(last_third) / len(last_third)
        
        if first_avg < avg_tension < last_avg:
            tension_profile = "Increasing tension"
        elif first_avg > avg_tension > last_avg:
            tension_profile = "Decreasing tension"
        elif max_tension in tension_scores[1:-1]:
            tension_profile = "Arch-shaped tension"
        else:
            tension_profile = "Consistent tension"
    
    return {
        'chords': chord_details,
        'average_tension': round(avg_tension, 2),
        'highest_tension': round(max_tension, 2),
        'progression_profile': tension_profile
    }

def analyze_inversions_in_dataframe(df, progression_column='progression', key_column='key_signature'):
    """
    Analyzes chord inversions and tension scores for all progressions in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions.
    progression_column : str, default='progression'
        The name of the column containing chord progressions.
    key_column : str, default='key_signature'
        The name of the column containing key signatures.
        
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with added chord inversion analysis columns.
    """
    import pandas as pd
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add new columns if they don't exist
    if 'average_tension' not in result_df.columns:
        result_df['average_tension'] = None
    if 'highest_tension' not in result_df.columns:
        result_df['highest_tension'] = None
    if 'tension_profile' not in result_df.columns:
        result_df['tension_profile'] = None
    
    # Process each row
    for idx, row in result_df.iterrows():
        # Get the progression and key signature
        progression = row[progression_column]
        key_sig = 'C'  # Default key
        
        if key_column in row.index and pd.notna(row[key_column]):
            key_sig = row[key_column]
        
        # Analyze the chord inversions
        analysis = analyze_chord_inversions(progression, key_sig)
        
        # Update the DataFrame
        result_df.at[idx, 'average_tension'] = analysis['average_tension']
        result_df.at[idx, 'highest_tension'] = analysis['highest_tension']
        result_df.at[idx, 'tension_profile'] = analysis['progression_profile']
        
        # Store detailed chord analysis as a serialized string
        chord_details = ', '.join([f"{c['roman_numeral']}({c['inversion_name']}, tension:{c['tension_score']})" 
                                  for c in analysis['chords'] if 'inversion_name' in c])
        result_df.at[idx, 'chord_details'] = chord_details
    
    return result_df

def create_progression_json_structure(df, progression_id=None) -> dict:
    """
    Creates a modular JSON structure from progression data, separating core data,
    metadata, and advanced insights.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing progression data
    progression_id : int or str, optional
        Specific progression ID to process. If None, processes all rows.
        
    Returns:
    --------
    dict
        A dictionary with modular organization:
        - core_data: Basic progression information
        - metadata: Version tracking and timestamps
        - musical_analysis: Cadence and harmonic analysis
        - audio_features: Spotify and audio-related data
        - advanced_insights: Tension profiles and recommendations
    """
    # Handle single progression or full dataset
    if progression_id is not None:
        df = df[df['progression_id'] == progression_id]
    
    result = []
    for _, row in df.iterrows():
        # Core progression data
        core_data = {
            'progression_id': row.get('progression_id'),
            'progression': row.get('progression'),
            'key_signature': row.get('key_signature'),
            'time_signature': row.get('time_signature', '4/4'),  # Default to common time
            'mode': row.get('mode', 'major')  # Default to major mode
        }
        
        # Metadata including version tracking
        metadata = {
            'version': row.get('version', 1),
            'created_at': row.get('created_at', str(datetime.now())),
            'last_updated': row.get('last_updated', str(datetime.now())),
            'song_title': row.get('song_title'),
            'artist': row.get('artist')
        }
        
        # Musical analysis data
        musical_analysis = {
            'cadence': {
                'type': row.get('cadence_type'),
                'strength': row.get('cadence_strength')
            },
            'harmonic_role': row.get('harmonic_role'),
            'chord_details': row.get('chord_details', '').split(', ') if row.get('chord_details') else []
        }
        
        # Audio features and Spotify data
        audio_features = {
            'spotify_id': row.get('spotify_id'),
            'tempo': row.get('tempo'),
            'energy': row.get('energy'),
            'loudness': row.get('loudness'),
            'danceability': row.get('danceability'),
            'valence': row.get('valence'),
            'instrumentalness': row.get('instrumentalness')
        }
        
        # Advanced musical insights
        advanced_insights = {
            'tension_profile': {
                'average_tension': row.get('average_tension'),
                'highest_tension': row.get('highest_tension'),
                'profile_type': row.get('tension_profile')
            },
            'pivot_chords': row.get('pivot_chords', set()),
            'recommended_modulations': row.get('recommended_pivot_chords', '').split(', ') if row.get('recommended_pivot_chords') else []
        }
        
        # Combine all sections
        progression_data = {
            'core_data': core_data,
            'metadata': metadata,
            'musical_analysis': musical_analysis,
            'audio_features': audio_features,
            'advanced_insights': advanced_insights
        }
        
        result.append(progression_data)
    
    # Return a single dict if querying specific progression_id, otherwise return list
    return result[0] if progression_id is not None and result else result

def convert_df_to_modular_json(df, output_file=None):
    """
    Converts a DataFrame to modular JSON format and optionally saves to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing progression data
    output_file : str, optional
        Path to save the JSON output. If None, returns the JSON string
        
    Returns:
    --------
    str or None
        JSON string if output_file is None, otherwise None
    """
    # Convert DataFrame to modular JSON structure
    json_data = create_progression_json_structure(df)
    
    # Convert to JSON string with proper formatting
    json_str = json.dumps(json_data, indent=2, default=str)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_str)
    else:
        return json_str

def analyze_progressions_in_batches(df, batch_size=1000, progression_column='progression', 
                                  key_column='key_signature', analysis_type='all'):
    """
    Processes chord progression analysis in batches to prevent memory overload.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions
    batch_size : int, default=1000
        Number of rows to process in each batch
    progression_column : str, default='progression'
        Name of the column containing chord progressions
    key_column : str, default='key_signature'
        Name of the column containing key signatures
    analysis_type : str, default='all'
        Type of analysis to perform: 'cadence', 'inversions', or 'all'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added analysis columns
    """
    # Validate input parameters
    if progression_column not in df.columns:
        raise ValueError(f"Column '{progression_column}' not found in DataFrame")
        
    # Initialize result DataFrame
    result_df = df.copy()
    
    # Initialize analysis columns based on analysis type
    if analysis_type in ['cadence', 'all']:
        if 'cadence_type' not in result_df.columns:
            result_df['cadence_type'] = None
        if 'cadence_strength' not in result_df.columns:
            result_df['cadence_strength'] = None
            
    if analysis_type in ['inversions', 'all']:
        if 'average_tension' not in result_df.columns:
            result_df['average_tension'] = None
        if 'highest_tension' not in result_df.columns:
            result_df['highest_tension'] = None
        if 'tension_profile' not in result_df.columns:
            result_df['tension_profile'] = None
        if 'chord_details' not in result_df.columns:
            result_df['chord_details'] = None
    
    # Calculate number of batches
    n_batches = int(np.ceil(len(df) / batch_size))
    
    # Process each batch
    for batch_num in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        try:
            # Process cadence analysis if requested
            if analysis_type in ['cadence', 'all']:
                batch_results = analyze_progressions_in_dataframe(
                    batch_df,
                    progression_column=progression_column,
                    key_column=key_column
                )
                result_df.loc[start_idx:end_idx-1, 'cadence_type'] = batch_results['cadence_type']
                result_df.loc[start_idx:end_idx-1, 'cadence_strength'] = batch_results['cadence_strength']
            
            # Process inversion analysis if requested
            if analysis_type in ['inversions', 'all']:
                batch_results = analyze_inversions_in_dataframe(
                    batch_df,
                    progression_column=progression_column,
                    key_column=key_column
                )
                result_df.loc[start_idx:end_idx-1, 'average_tension'] = batch_results['average_tension']
                result_df.loc[start_idx:end_idx-1, 'highest_tension'] = batch_results['highest_tension']
                result_df.loc[start_idx:end_idx-1, 'tension_profile'] = batch_results['tension_profile']
                result_df.loc[start_idx:end_idx-1, 'chord_details'] = batch_results['chord_details']
                
            # Clear music21 cache after each batch to prevent memory buildup
            from music21 import environment
            env = environment.Environment()
            env.purgeCache()
            
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {str(e)}")
            continue
    
    return result_df

def filter_advanced_progressions(df, filters=None, progression_column='progression', 
                              key_column='key_signature', mode_column='mode', 
                              min_tension=3.5, include_genres=None):
    """
    Filters chord progressions based on advanced musical characteristics including
    high-tension patterns, key changes, and genre-specific movement paths.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions
    filters : list of str, optional
        List of specific filters to apply: 'tension', 'key_changes', 'genre_patterns'
        If None, applies all filters
    progression_column : str, default='progression'
        Name of the column containing chord progressions
    key_column : str, default='key_signature'
        Name of the column containing key signatures
    mode_column : str, default='mode'
        Name of the column containing mode information
    min_tension : float, default=3.5
        Minimum tension score to consider for high-tension patterns
    include_genres : dict, optional
        Dictionary of genre names and their characteristic chord movements
        Example: {'jazz': ['ii-V-I', 'iii-vi-ii-V'], 'blues': ['I-IV-I-V']}
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only progressions matching the criteria
    """
    if filters is None:
        filters = ['tension', 'key_changes', 'genre_patterns']
        
    result_df = df.copy()
    original_length = len(result_df)
    
    # Define default genre patterns if none provided
    if include_genres is None:
        include_genres = {
            'jazz': [
                'ii-V-I', 'iii-vi-ii-V', 'i-VI-ii-V',  # Common jazz progressions
                'ii7-V7-Imaj7', 'iiø7-V7-i'  # Extended harmony patterns
            ],
            'blues': [
                'I-IV-I-V', 'I7-IV7-I7-V7',  # Basic blues changes
                'i7-iv7-i7-V7'  # Minor blues
            ],
            'modal': [
                'i-VII-VI',  # Dorian
                'i-vii-VI',  # Phrygian
                'I-II-vii'   # Lydian
            ]
        }
    
    # Filter for high-tension patterns
    if 'tension' in filters:
        # Get tension analysis
        tension_df = analyze_inversions_in_dataframe(result_df, 
                                                   progression_column=progression_column,
                                                   key_column=key_column)
        
        # Keep only high-tension progressions
        high_tension_mask = tension_df['average_tension'] >= min_tension
        result_df = result_df[high_tension_mask].copy()
        
        # Add tension scores to result
        result_df['tension_score'] = tension_df[high_tension_mask]['average_tension']
        result_df['tension_profile'] = tension_df[high_tension_mask]['tension_profile']
    
    # Filter for key changes and modulations
    if 'key_changes' in filters:
        def has_key_change(progression, key_sig):
            try:
                chords = progression.split('-')
                base_key = key.Key(key_sig.replace('m', '') if 'm' in key_sig else key_sig)
                
                for chord in chords:
                    rn = roman.RomanNumeral(chord, base_key)
                    # Check if chord suggests a different key area
                    if (rn.secondaryRomanNumeral or 
                        'applied' in rn.figure.lower() or 
                        any(acc in chord for acc in ['#', 'b'])):
                        return True
                return False
            except Exception:
                return False
        
        # Apply key change filter
        key_change_mask = result_df.apply(
            lambda row: has_key_change(row[progression_column], row[key_column]), 
            axis=1
        )
        result_df = result_df[key_change_mask].copy()
    
    # Filter for genre-specific patterns
    if 'genre_patterns' in filters:
        def matches_genre_pattern(progression):
            for genre, patterns in include_genres.items():
                if any(pattern in progression for pattern in patterns):
                    return genre
            return None
        
        # Apply genre pattern filter
        result_df['genre_match'] = result_df[progression_column].apply(matches_genre_pattern)
        result_df = result_df[result_df['genre_match'].notna()].copy()
    
    # Log filtering results
    print(f"Filtered {original_length - len(result_df)} progressions")
    print(f"Retained {len(result_df)} progressions matching criteria")
    
    return result_df

def batch_filter_progressions(df, batch_size=1000, **filter_kwargs):
    """
    Apply advanced progression filtering in batches to prevent memory overload.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing chord progressions
    batch_size : int, default=1000
        Size of batches to process
    **filter_kwargs : dict
        Keyword arguments to pass to filter_advanced_progressions
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame of all filtered results
    """
    # Calculate number of batches
    n_batches = int(np.ceil(len(df) / batch_size))
    filtered_results = []
    
    # Process each batch
    for batch_num in tqdm(range(n_batches), desc="Filtering progressions"):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        try:
            # Apply filtering to batch
            filtered_batch = filter_advanced_progressions(batch_df, **filter_kwargs)
            filtered_results.append(filtered_batch)
            
            # Clear music21 cache after each batch
            from music21 import environment
            env = environment.Environment()
            env.purgeCache()
            
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {str(e)}")
            continue
    
    # Combine results
    if filtered_results:
        return pd.concat(filtered_results, ignore_index=True)
    else:
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Test with some common progressions
    test_progressions = [
        {"progression": "I-IV-V-I", "key": "C"},
        {"progression": "ii-V-I", "key": "F"},
        {"progression": "I-V-vi", "key": "G"},
        {"progression": "IV-I", "key": "D"},
        {"progression": "i-iv6-V", "key": "Am"}
    ]
    
    print("Cadence Analysis Examples:")
    print("--------------------------")
    
    for test in test_progressions:
        prog = test["progression"]
        key_sig = test["key"]
        
        analysis = analyze_cadence(prog, key_sig)
        
        print(f"Progression: {prog} in {key_sig}")
        print(f"Cadence Type: {analysis['cadence_type']}")
        print(f"Strength: {analysis['cadence_strength']}")
        print(f"Analysis: {analysis['analysis']}")
        print()
    
    # Example with DataFrame
    import pandas as pd
    
    sample_data = {
        'progression': ['I-IV-V-I', 'ii-V-I', 'I-V-vi', 'IV-I', 'i-iv6-V'],
        'key_signature': ['C', 'F', 'G', 'D', 'Am'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("DataFrame Before Analysis:")
    print(df)
    
    df_with_analysis = analyze_progressions_in_dataframe(df)
    
    print("\nDataFrame After Analysis:")
    print(df_with_analysis[['progression', 'key_signature', 'cadence_type', 'cadence_strength']])
    
    # Test with inversions
    test_progressions_with_inversions = [
        {"progression": "I-IV-V6-I", "key": "C"},         # V in first inversion
        {"progression": "ii6-V-I", "key": "F"},           # ii in first inversion  
        {"progression": "I-V43-vi", "key": "G"},          # V7 in third inversion
        {"progression": "IV64-I", "key": "D"},            # IV in second inversion
        {"progression": "i-iv6-V7", "key": "Am"}          # iv in first inversion with dominant seventh
    ]
    
    print("\nChord Inversion Analysis Examples:")
    print("-----------------------------------")
    
    for test in test_progressions_with_inversions:
        prog = test["progression"]
        key_sig = test["key"]
        
        analysis = analyze_chord_inversions(prog, key_sig)
        
        print(f"Progression: {prog} in {key_sig}")
        print("Chord details:")
        for chord in analysis['chords']:
            inversion_info = chord.get('inversion_name', 'unknown')
            tension = chord.get('tension_score', 'N/A')
            print(f"  {chord['roman_numeral']}: {inversion_info}, tension score: {tension}")
        print(f"Average tension: {analysis['average_tension']}")
        print(f"Highest tension: {analysis['highest_tension']}")
        print(f"Progression profile: {analysis['progression_profile']}")
        print()
    
    # Example with DataFrame and inversions
    sample_data = {
        'progression': ['I-IV-V6-I', 'ii6-V-I', 'I-V43-vi', 'IV64-I', 'i-iv6-V7'],
        'key_signature': ['C', 'F', 'G', 'D', 'Am'],
        'song_title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("DataFrame Before Inversion Analysis:")
    print(df)
    
    df_with_analysis = analyze_inversions_in_dataframe(df)
    
    print("\nDataFrame After Inversion Analysis:")
    print(df_with_analysis[['progression', 'key_signature', 'average_tension', 'highest_tension', 'tension_profile']])