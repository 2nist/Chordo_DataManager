import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import logging
from requests.exceptions import RequestException, Timeout, ConnectionError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spotify_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("spotify_integration")

SPOTIFY_API_URL = "https://api.spotify.com/v1/"
TOKEN = "YOUR_SPOTIFY_API_TOKEN"  # Replace with your actual Spotify API token
MAX_RETRIES = 3
RETRY_DELAY = 2  # Base delay in seconds

def spotify_api_request(url, headers, params=None, max_retries=3, retry_delay=2):
    """
    Makes a Spotify API request with retry logic to handle rate limits, timeouts, and unexpected disconnections.

    Parameters:
    -----------
    url : str
        The Spotify API endpoint URL.
    headers : dict
        The headers to include in the request (e.g., Authorization).
    params : dict, optional
        Query parameters to include in the request.
    max_retries : int, default=3
        Maximum number of retries for the request.
    retry_delay : int, default=2
        Base delay in seconds between retries (exponential backoff).

    Returns:
    --------
    dict or None
        The JSON response from the API if successful, or None if all retries fail.
    """
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limiting
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                logger.warning(f"Rate limit reached. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            elif response.status_code in {500, 502, 503, 504}:  # Server errors
                logger.warning(f"Server error {response.status_code}. Retrying...")
            else:
                logger.error(f"Request failed with status code {response.status_code}: {response.text}")
                return None

        except Timeout:
            logger.warning("Request timed out. Retrying...")
        except ConnectionError:
            logger.warning("Connection error. Retrying...")
        except RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            return None

        retries += 1
        wait_time = retry_delay * (2 ** (retries - 1))  # Exponential backoff
        logger.info(f"Retrying in {wait_time} seconds...")
        time.sleep(wait_time)

    logger.error("Maximum retries exceeded. Request failed.")
    return None

def get_track_audio_features(track_id, retries=0):
    """
    Get audio features for a track from Spotify API with enhanced error handling
    
    Parameters:
    -----------
    track_id : str
        Spotify track ID
    retries : int
        Current retry count
        
    Returns:
    --------
    dict or None
        Audio features dictionary or None if unsuccessful
    """
    if retries >= MAX_RETRIES:
        logger.error(f"Maximum retries ({MAX_RETRIES}) exceeded for track ID: {track_id}")
        return None
        
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    
    try:
        response = requests.get(
            f"{SPOTIFY_API_URL}audio-features/{track_id}", 
            headers=headers,
            timeout=10  # Set a timeout of 10 seconds
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            logger.error("Unauthorized: Invalid or expired token. Please update your Spotify API token.")
            return None
        elif response.status_code == 429:  # Rate limiting
            retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * (retries + 1)))
            logger.warning(f"Rate limit reached. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return get_track_audio_features(track_id, retries + 1)
        elif response.status_code == 404:
            logger.warning(f"Track ID not found: {track_id}")
            return None
        else:
            logger.error(f"Error fetching audio features for track ID {track_id}: {response.status_code}")
            # Exponential backoff for other errors
            wait_time = RETRY_DELAY * (2 ** retries)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return get_track_audio_features(track_id, retries + 1)
            
    except Timeout:
        logger.warning(f"Request timed out for track ID {track_id}. Retrying...")
        wait_time = RETRY_DELAY * (2 ** retries)
        time.sleep(wait_time)
        return get_track_audio_features(track_id, retries + 1)
    except ConnectionError:
        logger.warning(f"Connection error for track ID {track_id}. Retrying...")
        wait_time = RETRY_DELAY * (2 ** retries)
        time.sleep(wait_time)
        return get_track_audio_features(track_id, retries + 1)
    except RequestException as e:
        logger.error(f"Request error for track ID {track_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for track ID {track_id}: {str(e)}")
        return None

def search_track(track_name, artist_name=None, retries=0):
    """
    Search for a track on Spotify with enhanced error handling
    
    Parameters:
    -----------
    track_name : str
        Name of the track to search for
    artist_name : str, optional
        Name of the artist
    retries : int
        Current retry count
        
    Returns:
    --------
    dict or None
        Track information or None if not found or error
    """
    if retries >= MAX_RETRIES:
        logger.error(f"Maximum retries ({MAX_RETRIES}) exceeded for track: {track_name}")
        return None
        
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    
    try:
        # Clean and format query parameters
        if not track_name or track_name.strip() == "":
            logger.warning("Empty track name provided")
            return None
            
        # URL encode query parameters to handle special characters
        if artist_name and artist_name.strip() != "":
            query = f"{track_name.strip()} artist:{artist_name.strip()}"
        else:
            query = track_name.strip()
        
        # Make the API request
        response = requests.get(
            f"{SPOTIFY_API_URL}search",
            headers=headers,
            params={"q": query, "type": "track", "limit": 1},
            timeout=10  # Set a timeout of 10 seconds
        )
        
        if response.status_code == 200:
            results = response.json()
            if results['tracks']['items']:
                return results['tracks']['items'][0]  # Return the first track found
            else:
                logger.info(f"No tracks found for '{track_name}'.")
                return None
        elif response.status_code == 401:
            logger.error("Unauthorized: Invalid or expired token. Please update your Spotify API token.")
            return None
        elif response.status_code == 429:  # Rate limiting
            retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * (retries + 1)))
            logger.warning(f"Rate limit reached. Waiting for {retry_after} seconds...")
            time.sleep(retry_after + 1)
            return search_track(track_name, artist_name, retries + 1)  # Retry after waiting
        else:
            logger.error(f"Error searching for track: {response.status_code}")
            # Exponential backoff for other errors
            wait_time = RETRY_DELAY * (2 ** retries)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return search_track(track_name, artist_name, retries + 1)
            
    except Timeout:
        logger.warning(f"Request timed out for track '{track_name}'. Retrying...")
        wait_time = RETRY_DELAY * (2 ** retries)
        time.sleep(wait_time)
        return search_track(track_name, artist_name, retries + 1)
    except ConnectionError:
        logger.warning(f"Connection error for track '{track_name}'. Retrying...")
        wait_time = RETRY_DELAY * (2 ** retries)
        time.sleep(wait_time)
        return search_track(track_name, artist_name, retries + 1)
    except RequestException as e:
        logger.error(f"Request error for track '{track_name}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for track '{track_name}': {str(e)}")
        return None

def enhance_df_with_spotify_data(df, song_title_column):
    """
    Query the Spotify API by song title and add artist, spotify_id, and song_title columns to the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to enhance with Spotify data
    song_title_column : str
        The name of the column containing song titles to search for
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with added artist, spotify_id, and song_title columns
    """
    # Create a copy of the dataframe to avoid modifying the original
    enhanced_df = df.copy()
    
    # Validate input DataFrame
    if song_title_column not in enhanced_df.columns:
        logger.error(f"Column '{song_title_column}' not found in DataFrame")
        return enhanced_df
    
    # Add new columns if they don't exist
    if 'spotify_id' not in enhanced_df.columns:
        enhanced_df['spotify_id'] = None
    if 'artist' not in enhanced_df.columns:
        enhanced_df['artist'] = None
    if 'song_title' not in enhanced_df.columns:
        enhanced_df['song_title'] = None
    if 'search_error' not in enhanced_df.columns:
        enhanced_df['search_error'] = None
    
    # Count stats for logging
    total_rows = len(enhanced_df)
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process each row
    logger.info(f"Querying Spotify API for {total_rows} tracks...")
    for idx, row in tqdm(enhanced_df.iterrows(), total=total_rows, desc="Searching tracks"):
        try:
            song_title = row[song_title_column]
            
            if pd.isna(song_title) or not song_title:
                enhanced_df.at[idx, 'search_error'] = 'Missing song title'
                skipped_count += 1
                continue
                
            # Check if we already have a spotify_id for this row
            if not pd.isna(row.get('spotify_id')) and row.get('spotify_id'):
                skipped_count += 1
                continue
                
            # Search for the track on Spotify
            track_info = search_track(song_title)
            
            if track_info:
                # Extract relevant information
                enhanced_df.at[idx, 'spotify_id'] = track_info['id']
                enhanced_df.at[idx, 'song_title'] = track_info['name']
                
                # Get artist information
                artists = ", ".join([artist['name'] for artist in track_info['artists']])
                enhanced_df.at[idx, 'artist'] = artists
                enhanced_df.at[idx, 'search_error'] = None
                success_count += 1
            else:
                # Track not found
                enhanced_df.at[idx, 'search_error'] = 'Track not found'
                error_count += 1
            
            # Rate limiting - avoid hitting Spotify's API limits
            time.sleep(0.2)
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = str(e)
            logger.error(f"Error processing row {idx}: {error_msg}")
            enhanced_df.at[idx, 'search_error'] = f'Error: {error_msg[:50]}'  # Truncate long error messages
            error_count += 1
    
    # Log summary statistics
    logger.info(f"Spotify search completed: {success_count} successful, {error_count} failed, {skipped_count} skipped.")
    
    return enhanced_df

def populate_audio_features(df, spotify_id_column='spotify_id'):
    """
    Retrieves tempo, time signature, and energy values from Spotify's audio features API
    and populates them in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the spotify_id column
    spotify_id_column : str, default='spotify_id'
        The name of the column containing Spotify track IDs
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with added tempo, time_signature, and energy columns
    """
    # Create a copy of the dataframe to avoid modifying the original
    enhanced_df = df.copy()
    
    # Validate input DataFrame
    if spotify_id_column not in enhanced_df.columns:
        logger.error(f"Column '{spotify_id_column}' not found in DataFrame")
        return enhanced_df
    
    # Add new columns if they don't exist
    if 'tempo' not in enhanced_df.columns:
        enhanced_df['tempo'] = None
    if 'time_signature' not in enhanced_df.columns:
        enhanced_df['time_signature'] = None
    if 'energy' not in enhanced_df.columns:
        enhanced_df['energy'] = None
    if 'features_error' not in enhanced_df.columns:
        enhanced_df['features_error'] = None
    
    # Count stats for logging
    total_rows = len(enhanced_df)
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process each row
    logger.info(f"Fetching audio features for {total_rows} tracks...")
    for idx, row in tqdm(enhanced_df.iterrows(), total=total_rows, desc="Fetching features"):
        try:
            spotify_id = row[spotify_id_column]
            
            if pd.isna(spotify_id) or not spotify_id:
                enhanced_df.at[idx, 'features_error'] = 'Missing Spotify ID'
                skipped_count += 1
                continue
                
            # Check if we already have features for this row
            if not pd.isna(row.get('tempo')) and not pd.isna(row.get('time_signature')) and not pd.isna(row.get('energy')):
                skipped_count += 1
                continue
                
            # Get audio features from Spotify API
            audio_features = get_track_audio_features(spotify_id)
            
            if audio_features:
                # Extract relevant audio features
                enhanced_df.at[idx, 'tempo'] = audio_features.get('tempo')
                enhanced_df.at[idx, 'time_signature'] = audio_features.get('time_signature')
                enhanced_df.at[idx, 'energy'] = audio_features.get('energy')
                enhanced_df.at[idx, 'features_error'] = None
                success_count += 1
            else:
                # Features not available
                enhanced_df.at[idx, 'features_error'] = 'Features not available'
                error_count += 1
            
            # Rate limiting - avoid hitting Spotify's API limits
            time.sleep(0.2)
                
        except Exception as e:
            # Handle unexpected errors
            error_msg = str(e)
            logger.error(f"Error processing features for row {idx}: {error_msg}")
            enhanced_df.at[idx, 'features_error'] = f'Error: {error_msg[:50]}'  # Truncate long error messages
            error_count += 1
    
    # Log summary statistics
    logger.info(f"Features retrieval completed: {success_count} successful, {error_count} failed, {skipped_count} skipped.")
    
    return enhanced_df

def get_and_populate_audio_features(df, song_title_column):
    """
    Convenience function that combines enhance_df_with_spotify_data and populate_audio_features.
    First adds spotify_id and then retrieves audio features in one step.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing song titles
    song_title_column : str
        The name of the column containing song titles
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with added spotify_id, artist, song_title, tempo, time_signature, and energy columns
    """
    try:
        # First get the spotify IDs
        logger.info("Step 1: Getting Spotify IDs for tracks...")
        enhanced_df = enhance_df_with_spotify_data(df, song_title_column)
        
        # Then get the audio features
        logger.info("Step 2: Getting audio features...")
        final_df = populate_audio_features(enhanced_df)
        
        # Generate a simple report on the success rate
        total_rows = len(final_df)
        id_success = final_df['spotify_id'].notna().sum()
        features_success = final_df['tempo'].notna().sum()
        
        logger.info(f"Process complete. Retrieved {id_success}/{total_rows} Spotify IDs ({id_success/total_rows:.1%})")
        logger.info(f"Retrieved audio features for {features_success}/{total_rows} tracks ({features_success/total_rows:.1%})")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error in get_and_populate_audio_features: {str(e)}")
        # Return the original dataframe if there was an error
        return df

def fetch_spotify_data():
    """
    Placeholder function referenced in main.py
    This would typically load a DataFrame and call enhance_df_with_spotify_data
    """
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        data_path = "../data/cleaned_chordonomicon.parquet"
        
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"Successfully loaded dataset with {len(df)} rows.")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None
        
        # Get Spotify data
        enhanced_df = get_and_populate_audio_features(df, 'song_title')
        
        # Save the enhanced dataset
        try:
            output_path = "../data/enhanced_chordonomicon.parquet"
            enhanced_df.to_parquet(output_path)
            logger.info(f"Enhanced dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving enhanced dataset: {str(e)}")
        
        return enhanced_df
        
    except Exception as e:
        logger.error(f"Unexpected error in fetch_spotify_data: {str(e)}")
        return None

def handle_missing_spotify_data(df, columns_to_check, default_value='Unknown'):
    """
    Checks for None values in specified columns and assigns a default value.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to process.
    columns_to_check : list of str
        List of column names to check for None values.
    default_value : str, default='Unknown'
        The value to assign to missing entries.

    Returns:
    --------
    pandas.DataFrame
        The updated DataFrame with missing values handled.
    """
    for column in columns_to_check:
        if column in df.columns:
            df[column] = df[column].fillna(default_value)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")
    return df

def validate_audio_features(df, tempo_column='tempo', energy_column='energy'):
    """
    Validates that tempo values fall between 40 and 240 BPM, and energy values are between 0 and 1.0.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to validate.
    tempo_column : str, default='tempo'
        The name of the column containing tempo values.
    energy_column : str, default='energy'
        The name of the column containing energy values.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with invalid rows removed.
    """
    if tempo_column in df.columns:
        invalid_tempo = ~df[tempo_column].between(40, 240, inclusive="both")
        if invalid_tempo.any():
            logger.warning(f"Removing {invalid_tempo.sum()} rows with invalid tempo values.")
        df = df[~invalid_tempo]
    else:
        logger.warning(f"Tempo column '{tempo_column}' not found in DataFrame.")

    if energy_column in df.columns:
        invalid_energy = ~df[energy_column].between(0, 1.0, inclusive="both")
        if invalid_energy.any():
            logger.warning(f"Removing {invalid_energy.sum()} rows with invalid energy values.")
        df = df[~invalid_energy]
    else:
        logger.warning(f"Energy column '{energy_column}' not found in DataFrame.")

    return df

# Example usage
if __name__ == "__main__":
    # Example 1: Search for a track and get audio features
    track = search_track("Shape of You", "Ed Sheeran")
    if track:
        track_id = track['id']
        audio_features = get_track_audio_features(track_id)
        print(json.dumps(audio_features, indent=2))
    
    # Example 2: Enhance a DataFrame with Spotify data
    sample_data = {
        'title': ['Shape of You', 'Bohemian Rhapsody', 'Hotel California']
    }
    df = pd.DataFrame(sample_data)
    
    enhanced_df = enhance_df_with_spotify_data(df, 'title')
    print("\nEnhanced DataFrame:")
    print(enhanced_df)
    
    # Example 3: Retrieve audio features for a DataFrame with Spotify IDs
    if track:
        sample_data = {
            'title': ['Shape of You'],
            'spotify_id': [track['id']]
        }
        df_with_ids = pd.DataFrame(sample_data)
        
        df_with_features = populate_audio_features(df_with_ids)
        print("\nDataFrame with Audio Features:")
        print(df_with_features)
        
    # Example 4: Get Spotify IDs and audio features in one step
    sample_data = {
        'title': ['Shape of You', 'Bohemian Rhapsody', 'Hotel California']
    }
    df = pd.DataFrame(sample_data)
    
    complete_df = get_and_populate_audio_features(df, 'title')
    print("\nComplete DataFrame with Audio Features:")
    print(complete_df)

    # Example 5: Handle missing Spotify data
    sample_data = {
        'spotify_id': ['123', None, '456'],
        'artist': ['Artist A', None, 'Artist C'],
        'song_title': ['Song A', 'Song B', None]
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Handle missing Spotify data
    updated_df = handle_missing_spotify_data(df, ['spotify_id', 'artist', 'song_title'])

    print("\nUpdated DataFrame:")
    print(updated_df)

    # Example Spotify API request
    example_url = f"{SPOTIFY_API_URL}search"
    example_headers = {"Authorization": f"Bearer {TOKEN}"}
    example_params = {"q": "Shape of You", "type": "track", "limit": 1}

    response = spotify_api_request(example_url, example_headers, example_params)
    if response:
        print("API Response:", response)
    else:
        print("API request failed.")

    # Example DataFrame
    sample_data = {
        'tempo': [120, 300, 80, 35],
        'energy': [0.8, 1.2, 0.5, -0.1],
        'other_info': ['Info1', 'Info2', 'Info3', 'Info4']
    }
    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)

    # Validate the DataFrame
    validated_df = validate_audio_features(df)

    print("\nValidated DataFrame:")
    print(validated_df)