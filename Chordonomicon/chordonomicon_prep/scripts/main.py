import clean_data
import spotify_integration
import audio_features

def main():
    # Step 1: Clean the original dataset
    print("Cleaning the dataset...")
    clean_data.clean_dataset()

    # Step 2: Integrate with Spotify to fetch audio features
    print("Integrating with Spotify...")
    spotify_data = spotify_integration.fetch_spotify_data()

    # Step 3: Process audio features
    print("Processing audio features...")
    audio_features.process_audio_features(spotify_data)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()