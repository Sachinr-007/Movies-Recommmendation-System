from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__)

# Global variables for dataset
df_clean = None
all_genres = []

def find_csv_file():
    """Find the CSV file in various possible locations"""
    possible_paths = [
        'imdb-movies-dataset.csv',
        './imdb-movies-dataset.csv',

        os.path.join(os.path.dirname(__file__), 'imdb-movies-dataset.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb-movies-dataset.csv'),
    ]
    
    # Also check current working directory
    cwd = os.getcwd()
    possible_paths.append(os.path.join(cwd, 'imdb-movies-dataset.csv'))
    
    print("\n" + "="*80)
    print("Searching for dataset file...")
    print("="*80)
    print(f"Current working directory: {cwd}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("\nChecking paths:")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {'Yes' if exists else 'No'} {abs_path}")
        if exists:
            print(f"\n Found dataset at: {abs_path}")
            return path
    
    print("\n Dataset file not found in any location!")
    print("\n Please ensure 'imdb-movies-dataset.csv' is in one of these locations:")
    
    # List files in current directory
    print("\nFiles in current directory:")
    try:
        files = os.listdir('.')
        for f in files[:20]:  # For Showing first 20 files
            print(f"  - {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    return None


def load_dataset():
    """Load and prepare the movie dataset"""
    global df_clean, all_genres
    
    try:
        # Finding CSV file
        csv_path = find_csv_file()
        
        if csv_path is None:
            return False
        
        print("\n" + "="*80)
        print("Loading dataset...")
        print("="*80)
        
        # Loading dataset with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("Trying different encoding...")
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
        
        # Checking required columns
        required_cols = ['Title', 'Genre']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Cleaning dataset
        df_clean = df.dropna(subset=['Title', 'Genre']).copy()
        print(f" After cleaning: {len(df_clean)} movies with valid Title and Genre")
        
        # Parsing genres
        df_clean['Genre_List'] = df_clean['Genre'].apply(
            lambda x: [g.strip() for g in str(x).split(',')]
        )
        
        # Geting all unique genres
        all_genres_set = set()
        for genre_list in df_clean['Genre_List']:
            all_genres_set.update(genre_list)
        all_genres = sorted(list(all_genres_set))
        
        print(f" Found {len(all_genres)} unique genres")
        print(f"  Genres: {', '.join(all_genres[:10])}{'...' if len(all_genres) > 10 else ''}")
        
        # Show sample data
        print("\n Sample movie data:")
        sample = df_clean.head(3)
        for idx, row in sample.iterrows():
            print(f"  - {row['Title']} ({row.get('Year', 'N/A')}) - {row['Genre']}")
        
        print("\n" + "="*80)
        print("Dataset loaded successfully!")
        print("="*80 + "\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n Error: File not found - {e}")
        return False
    except pd.errors.EmptyDataError:
        print("\n Error: CSV file is empty")
        return False
    except Exception as e:
        print(f"\n Error loading dataset: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_genre_overlap(genres1, genres2):
    """Calculate similarity between two genre lists"""
    set1 = set(genres1)
    set2 = set(genres2)
    
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def search_by_movie_name(movie_name, num_results=5):
    """Find similar movies based on movie name"""
    
    try:
        # Finding movie (case-insensitive partial match)
        matches = df_clean[df_clean['Title'].str.contains(movie_name, case=False, na=False, regex=False)]
        
        if len(matches) == 0:
            return None, f"Movie '{movie_name}' not found!"
        
        # Use first match
        target = matches.iloc[0]
        target_genres = target['Genre_List']
        
        print(f"Found target movie: {target['Title']} with genres: {target_genres}")
        
        # Find similar movies
        df_temp = df_clean.copy()
        df_temp['Similarity'] = df_temp['Genre_List'].apply(
            lambda x: calculate_genre_overlap(target_genres, x)
        )
        
        # Exclude the target movie
        candidates = df_temp[df_temp['Title'] != target['Title']].copy()
        candidates = candidates[candidates['Similarity'] > 0]
        
        if len(candidates) == 0:
            return None, "No similar movies found!"
        
        # Add rating score for better ranking
        candidates['Rating_Score'] = pd.to_numeric(candidates.get('Rating', 0), errors='coerce').fillna(0)
        candidates['Final_Score'] = candidates['Similarity'] * 0.7 + (candidates['Rating_Score'] / 10) * 0.3
        
        # Get top results
        results = candidates.nlargest(num_results, 'Final_Score')
        
        print(f"Found {len(results)} similar movies")
        
        return results, None
        
    except Exception as e:
        print(f"Error in search_by_movie_name: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Search error: {str(e)}"


def search_by_genre(genre_input, num_results=5, min_rating=0.0):
    """Find movies by genre(s)"""
    
    # Parse genres
    search_genres = [g.strip().title() for g in genre_input.split(',')]
    
    # Validate genres
    valid_genres = [g for g in search_genres if g in all_genres]
    
    if not valid_genres:
        return None, f"No valid genres found. Available genres: {', '.join(all_genres[:10])}"
    
    # Find movies with matching genres
    def has_any_genre(movie_genres):
        return any(g in movie_genres for g in valid_genres)
    
    candidates = df_clean[df_clean['Genre_List'].apply(has_any_genre)].copy()
    
    if len(candidates) == 0:
        return None, "No movies found for these genres!"
    
    # Calculate match score
    candidates['Match_Score'] = candidates['Genre_List'].apply(
        lambda x: calculate_genre_overlap(valid_genres, x)
    )
    
    # Apply rating filter
    candidates['Rating_Num'] = pd.to_numeric(candidates.get('Rating', 0), errors='coerce').fillna(0)
    candidates = candidates[candidates['Rating_Num'] >= min_rating]
    
    if len(candidates) == 0:
        return None, f"No movies found with rating >= {min_rating}"
    
    # Combined score
    candidates['Final_Score'] = candidates['Match_Score'] * 0.6 + (candidates['Rating_Num'] / 10) * 0.4
    
    # Get top results
    results = candidates.nlargest(num_results, 'Final_Score')
    
    return results, None


def smart_search(query, num_results=5, min_rating=0.0):
    """Smart search that detects if input is movie name or genre"""
    
    query = query.strip()
    
    # Check if query matches a genre
    query_genres = [g.strip().title() for g in query.split(',')]
    is_genre_search = any(g in all_genres for g in query_genres)
    
    # Check if query matches a movie name
    movie_matches = df_clean[df_clean['Title'].str.contains(query, case=False, na=False)]
    is_movie_search = len(movie_matches) > 0
    
    # Determine search type
    if is_genre_search and not is_movie_search:
        # Pure genre search
        return search_by_genre(query, num_results, min_rating)
    elif is_movie_search:
        # Movie name search (prioritize over genre)
        return search_by_movie_name(query, num_results)
    else:
        # Try movie name first
        return search_by_movie_name(query, num_results)


@app.route('/')
def index():
    """Serve the main page"""
    if df_clean is None:
        return """
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; padding: 50px; background: #141414; color: white;">
            <h1 style="color: #e50914;"> Dataset Not Loaded</h1>
            <p>The movie dataset could not be loaded. Please check the console for error messages.</p>
            <p>Make sure 'imdb-movies-dataset.csv' is in the same directory as app.py</p>
            <a href="/" style="color: #e50914;">Retry</a>
        </body>
        </html>
        """, 500
    
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for movie search"""
    
    if df_clean is None:
        return jsonify({'error': 'Dataset not loaded. Please restart the server.'}), 500
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        num_results = int(data.get('num_results', 5))
        min_rating = float(data.get('min_rating', 0.0))
        
        # Validate input
        if len(query) < 2:
            return jsonify({'error': 'Please enter at least 2 characters'}), 400
        
        # Clamp values
        num_results = max(3, min(num_results, 10))
        min_rating = max(0.0, min(min_rating, 10.0))
        
        # Perform search
        results, error = smart_search(query, num_results, min_rating)
        
        if error:
            return jsonify({'error': error}), 404
        
        # Convert results to JSON
        movies = []
        for _, movie in results.iterrows():
            movies.append({
                'Title': str(movie['Title']),
                'Year': str(movie.get('Year', 'N/A')),
                'Rating': str(movie.get('Rating', 'N/A')),
                'Genre': str(movie.get('Genre', '')),
                'Director': str(movie.get('Director', 'N/A')),
                'Cast': str(movie.get('Cast', 'N/A')),
                'Description': str(movie.get('Description', '')) if pd.notna(movie.get('Description')) else ''
            })
        
        return jsonify({'results': movies})
        
    except Exception as e:
        print(f"Error in search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Search error: {str(e)}'}), 500


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """API endpoint to get all available genres"""
    if df_clean is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    return jsonify({'genres': all_genres})


@app.route('/api/status', methods=['GET'])
def status():
    """Check if dataset is loaded"""
    return jsonify({
        'loaded': df_clean is not None,
        'movies': len(df_clean) if df_clean is not None else 0,
        'genres': len(all_genres)
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MOVIE RECOMMENDATION SYSTEM - WEB APP")
    print("="*80 + "\n")
    
    # Load dataset on startup
    if load_dataset():
        print(" Server starting...")
        print(f" Access the app at: http://127.0.0.1:5000")
        print(f" Or from network: http://0.0.0.0:5000")
        print("\nPress CTRL+C to stop the server")
        print("="*80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*80)
        print(" FAILED TO START")
        print("="*80)
        print("\nPlease fix the dataset issue and try again.")
        print("\nTroubleshooting:")
        print("  1. Verify the CSV file exists")
        print("  2. Check the file name (must be: imdb-movies-dataset.csv)")
        print("  3. Ensure the file is not corrupted")
        print("  4. Place the file in the same directory as app.py")
        sys.exit(1)