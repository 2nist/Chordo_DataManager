import clean_data
import spotify_integration
import audio_features
import argparse
from api_server import app as api_app

def process_data():
    """Run the full data processing pipeline"""
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

def start_api_server(host='0.0.0.0', port=5000, debug=True):
    """Start the API server"""
    print(f"Starting API server on {host}:{port}...")
    api_app.run(host=host, port=port, debug=debug)

def main():
    parser = argparse.ArgumentParser(description='Chordonomicon Dataset Manager')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Run the data processing pipeline')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_data()
    elif args.command == 'api':
        start_api_server(args.host, args.port, args.debug)
    else:
        # Default to process data if no command is specified
        process_data()

if __name__ == "__main__":
    main()