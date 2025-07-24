"""
📜 Bible Verse Vault Builder 🧘‍♂️💾
This script loads, formats, embeds, and stores verses from the KJV Bible using ChromaDB for semantic search.
It ensures GPU support (if available) and batches embeddings to stay within ChromaDB's limits.

Requirements:
- pandas
- sentence-transformers
- chromadb
- torch (for GPU support)
"""

import pandas as pd
import torch
import logging
import os
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =======================
# CONFIGURATION
# =======================

# Define your lookup dictionary for book IDs to book names
BOOK_LOOKUP: Dict[int, str] = {
    1: "Genesis", 2: "Exodus", 3: "Leviticus", 4: "Numbers", 5: "Deuteronomy",
    6: "Joshua", 7: "Judges", 8: "Ruth", 9: "1 Samuel", 10: "2 Samuel",
    11: "1 Kings", 12: "2 Kings", 13: "1 Chronicles", 14: "2 Chronicles",
    15: "Ezra", 16: "Nehemiah", 17: "Esther", 18: "Job", 19: "Psalms",
    20: "Proverbs", 21: "Ecclesiastes", 22: "Song of Solomon", 23: "Isaiah",
    24: "Jeremiah", 25: "Lamentations", 26: "Ezekiel", 27: "Daniel",
    28: "Hosea", 29: "Joel", 30: "Amos", 31: "Obadiah", 32: "Jonah",
    33: "Micah", 34: "Nahum", 35: "Habakkuk", 36: "Zephaniah", 37: "Haggai",
    38: "Zechariah", 39: "Malachi", 40: "Matthew", 41: "Mark", 42: "Luke",
    43: "John", 44: "Acts", 45: "Romans", 46: "1 Corinthians", 47: "2 Corinthians",
    48: "Galatians", 49: "Ephesians", 50: "Philippians", 51: "Colossians",
    52: "1 Thessalonians", 53: "2 Thessalonians", 54: "1 Timothy", 55: "2 Timothy",
    56: "Titus", 57: "Philemon", 58: "Hebrews", 59: "James", 60: "1 Peter",
    61: "2 Peter", 62: "1 John", 63: "2 John", 64: "3 John", 65: "Jude", 66: "Revelation"
}

# Configuration constants
CSV_PATH = "KJV_Verses.csv"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "bible_verses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000  # Reduced for better memory management

# =======================
# STEP 1: Load and Validate Bible Verses
# =======================

def load_and_format_verses(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load Bible verses from CSV and format them for embedding.
    
    Args:
        csv_path: Path to the CSV file containing Bible verses
        
    Returns:
        DataFrame with formatted verses or None if loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return None
        
        # Load CSV and clean column headers
        logger.info(f"Loading Bible verses from {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Remove any whitespace from column names
        
        # Validate required columns exist
        required_columns = ['book_id', 'chapter', 'verse', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Remove any rows with missing data
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with missing data")
        
        # Format verses with readable structure: "Book Chapter:Verse — Text"
        df["combined"] = df.apply(
            lambda row: f"{BOOK_LOOKUP.get(row['book_id'], 'Unknown')} {row['chapter']}:{row['verse']} — {row['text']}",
            axis=1
        )
        
        logger.info(f"Successfully loaded and formatted {len(df)} verses")
        
        # Show preview of first few verses
        logger.info("Preview of formatted verses:")
        for i, verse in enumerate(df["combined"].head(3)):
            logger.info(f"  {i+1}. {verse[:100]}...")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

# =======================
# STEP 2: Generate Embeddings with Sentence Transformers
# =======================

def setup_embedding_model(model_name: str) -> Optional[SentenceTransformer]:
    """
    Initialize the sentence transformer model with GPU support if available.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Initialized SentenceTransformer model or None if setup fails
    """
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load the sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        
        logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
        
    except Exception as e:
        logger.error(f"Error setting up embedding model: {e}")
        return None

def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        model: Initialized SentenceTransformer model
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors or None if generation fails
    """
    try:
        logger.info(f"Generating embeddings for {len(texts)} verses...")
        
        # Convert verse strings into high-dimensional vectors
        embeddings = model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
        return embeddings.tolist()  # Convert to list for ChromaDB compatibility
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

# =======================
# STEP 3: Store Embeddings in ChromaDB
# =======================

def setup_chromadb(db_path: str, collection_name: str):
    """
    Initialize ChromaDB client and collection.
    
    Args:
        db_path: Path where ChromaDB will store data
        collection_name: Name of the collection to create/get
        
    Returns:
        Tuple of (client, collection) or (None, None) if setup fails
    """
    try:
        # Initialize Chroma client with persistent storage
        logger.info(f"Initializing ChromaDB at {db_path}")
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get existing collection
        # This won't fail if collection already exists
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "KJV Bible verses with semantic embeddings"}
        )
        
        # Check if collection already has data
        existing_count = collection.count()
        if existing_count > 0:
            logger.info(f"Found existing collection with {existing_count} verses")
        else:
            logger.info("Created new empty collection")
        
        return client, collection
        
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        return None, None

def store_embeddings_in_batches(collection, df: pd.DataFrame, embeddings: List[List[float]], batch_size: int = 1000):
    """
    Store embeddings in ChromaDB in manageable batches.
    
    Args:
        collection: ChromaDB collection object
        df: DataFrame containing verse data
        embeddings: List of embedding vectors
        batch_size: Number of embeddings to process per batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        total_verses = len(df)
        logger.info(f"Storing {total_verses} verses in batches of {batch_size}")
        
        # Clear existing data if any (optional - remove if you want to append)
        existing_count = collection.count()
        if existing_count > 0:
            logger.info(f"Clearing existing {existing_count} entries from collection")
            collection.delete(where={})  # Delete all existing entries
        
        # Process in batches to avoid memory issues and ChromaDB limits
        successful_batches = 0
        for i in range(0, total_verses, batch_size):
            end = min(i + batch_size, total_verses)
            batch_num = i // batch_size + 1
            total_batches = (total_verses + batch_size - 1) // batch_size
            
            try:
                # Prepare batch data
                batch_documents = df["combined"].tolist()[i:end]
                batch_embeddings = embeddings[i:end]
                batch_ids = [str(j) for j in df.index[i:end]]
                batch_metadatas = df[["book_id", "chapter", "verse"]].to_dict(orient="records")[i:end]
                
                # Add batch to collection
                collection.add(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                
                successful_batches += 1
                logger.info(f"Successfully added batch {batch_num}/{total_batches} ({end - i} verses)")
                
            except Exception as e:
                logger.error(f"Error adding batch {batch_num}: {e}")
                continue
        
        # Verify final count
        final_count = collection.count()
        logger.info(f"Storage complete. Total verses in collection: {final_count}")
        
        return successful_batches > 0
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        return False

# =======================
# STEP 4: Semantic Search Functions
# =======================

def semantic_search(collection, query: str, n_results: int = 5) -> Optional[List[Dict]]:
    """
    Perform semantic search on the Bible verse collection.
    
    Args:
        collection: ChromaDB collection object
        query: Search query string
        n_results: Number of results to return
        
    Returns:
        List of search results or None if search fails
    """
    try:
        logger.info(f"Searching for: '{query}'")
        
        # Run semantic query against the verse collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results for better readability
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else None
            
            # Get book name from lookup
            book_name = BOOK_LOOKUP.get(int(metadata['book_id']), 'Unknown')
            verse_reference = f"{book_name} {metadata['chapter']}:{metadata['verse']}"
            
            formatted_results.append({
                'reference': verse_reference,
                'text': doc,
                'distance': distance,
                'metadata': metadata
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        return None

def display_search_results(results: List[Dict]):
    """
    Display search results in a formatted way.
    
    Args:
        results: List of search result dictionaries
    """
    if not results:
        logger.info("No results found.")
        return
    
    logger.info(f"Found {len(results)} relevant verses:")
    print("\n" + "="*80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['reference']}")
        print(f"   {result['text']}")
        if result['distance'] is not None:
            print(f"   (Similarity score: {1 - result['distance']:.3f})")
        print()

# =======================
# MAIN EXECUTION
# =======================

def main():
    """
    Main function to orchestrate the Bible verse vault building process.
    """
    logger.info("🚀 Starting Bible Verse Vault Builder")
    
    # Step 1: Load and format Bible verses
    df = load_and_format_verses(CSV_PATH)
    if df is None:
        logger.error("Failed to load Bible verses. Exiting.")
        return
    
    # Step 2: Setup embedding model
    model = setup_embedding_model(EMBEDDING_MODEL)
    if model is None:
        logger.error("Failed to setup embedding model. Exiting.")
        return
    
    # Step 3: Generate embeddings
    embeddings = generate_embeddings(model, df["combined"].tolist())
    if embeddings is None:
        logger.error("Failed to generate embeddings. Exiting.")
        return
    
    # Step 4: Setup ChromaDB
    client, collection = setup_chromadb(CHROMA_DB_PATH, COLLECTION_NAME)
    if client is None or collection is None:
        logger.error("Failed to setup ChromaDB. Exiting.")
        return
    
    # Step 5: Store embeddings
    success = store_embeddings_in_batches(collection, df, embeddings, BATCH_SIZE)
    if not success:
        logger.error("Failed to store embeddings. Exiting.")
        return
    
    # Step 6: Test semantic search
    logger.info("🔍 Testing semantic search functionality")
    
    # Sample queries to test the system
    test_queries = [
        "What does the Bible say about forgiveness?",
        "Love your enemies",
        "Faith and hope",
        "God's mercy and grace"
    ]
    
    for query in test_queries:
        results = semantic_search(collection, query, n_results=3)
        if results:
            print(f"\n🔎 Query: '{query}'")
            display_search_results(results)
        else:
            logger.error(f"Search failed for query: '{query}'")
    
    logger.info("✅ Bible Verse Vault Builder completed successfully!")
    
    # Display usage instructions
    print("\n" + "="*80)
    print("📖 Your Bible Verse Vault is ready!")
    print("You can now perform semantic searches using the collection object.")
    print("Example usage:")
    print("  results = semantic_search(collection, 'your search query', n_results=5)")
    print("  display_search_results(results)")
    print("="*80)

if __name__ == "__main__":
    main()
