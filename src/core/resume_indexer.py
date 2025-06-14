"""
Resume Indexer
-------------
A tool for processing resumes (PDF/DOCX), generating embeddings, and storing them in Pinecone.
"""

# ----------------------------
# Imports
# ----------------------------
# Standard library imports
import os
import logging
import time
import traceback
import math
import threading
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import docx
import pdfplumber
from tqdm import tqdm
import huggingface_hub
from huggingface_hub.constants import HF_HUB_DOWNLOAD_TIMEOUT
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Resume indexing configuration."""
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "resume-index")
    MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    DEVICE: Optional[str] = None
    MAX_TOKENS_PER_CHUNK: int = 384
    OVERLAP_TOKENS: int = 50
    BATCH_SIZE: int = 32
    DATA_FOLDER: str = "data"
    MODEL_CACHE_DIR: str = "models"
    CACHE_TTL: int = 3600
    def __post_init__(self):
        """Validate config and create directories."""
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set")
        os.makedirs(self.DATA_FOLDER, exist_ok=True)
        os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)
        if "large" in self.MODEL_NAME:
            self.MAX_TOKENS_PER_CHUNK = 512
            self.OVERLAP_TOKENS = 64
        else:
            self.MAX_TOKENS_PER_CHUNK = 384
            self.OVERLAP_TOKENS = 50

def download_nltk_resources():
    """Ensure required NLTK resources are available."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
                raise

download_nltk_resources()

class ResumeProcessor:
    """Resume text extraction and chunking."""
    def __init__(self, tokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config = config
        self._text_cache: Dict[int, List[str]] = {}

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            logger.error(f"Error extracting text from DOCX file {file_path}: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        try:
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                text_parts = [
                    page.extract_text() + "\n"
                    for page in pdf.pages
                    if page.extract_text()
                ]
            return "".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF file {file_path}: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        try:
            cache_key = hash(text)
            if cache_key in self._text_cache:
                return self._text_cache[cache_key]
            tokens = self.tokenizer.encode(text)
            step = max(1, self.config.MAX_TOKENS_PER_CHUNK - self.config.OVERLAP_TOKENS)
            chunk_boundaries = range(0, len(tokens), step)
            chunks = [
                self.tokenizer.decode(tokens[i:min(i + self.config.MAX_TOKENS_PER_CHUNK, len(tokens))])
                for i in chunk_boundaries
            ]
            self._text_cache[cache_key] = chunks
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def clear_cache(self) -> None:
        self._text_cache.clear()

class Embedder:
    """Handles text embedding using BGE model."""
    
    def __init__(self, config: Config):
        """Initialize the embedder with configuration."""
        try:
            self.config = config  # Store config object
            logger.info(f"Loading embedding model: {config.MODEL_NAME}")
            
            # Determine device
            if config.DEVICE is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.DEVICE
            
            # Increase timeout for model downloads
            huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_NAME,
                trust_remote_code=True,
                cache_dir=config.MODEL_CACHE_DIR,
                local_files_only=False,
                timeout=300
            )
            
            # Load model
            self.model = SentenceTransformer(
                config.MODEL_NAME,
                device=self.device,
                cache_folder=config.MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model or tokenizer: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

    def embed(self, text: Union[str, List[str]], batch_size: int = None) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text with batching support."""
        try:
            if batch_size is None:
                batch_size = self.config.BATCH_SIZE
                
            is_single_input = isinstance(text, str)
            if is_single_input:
                text = [text]
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(text), batch_size):
                batch_text = text[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_text,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                all_embeddings.extend(batch_embeddings)
            
            if not hasattr(all_embeddings[0], 'tolist'):
                raise ValueError(f"Unexpected embedding type: {type(all_embeddings[0])}")
            
            return all_embeddings[0].tolist() if is_single_input else [e.tolist() for e in all_embeddings]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

# Vector Database Management
class PineconeManager:
    """Manages interactions with Pinecone vector database."""
    
    def __init__(self, config: Config):
        """Initialize Pinecone connection."""
        try:
            logger.info("Initializing Pinecone connection...")
            logger.info(f"Using Pinecone index: {config.PINECONE_INDEX_NAME}")
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            self._cache: Dict[str, Any] = {}
            self._cache_ttl = config.CACHE_TTL
            logger.info("Pinecone connection established")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            raise

    def store_vectors(self, 
                     embeddings: List[List[float]],
                     chunks: List[str], 
                     candidate_id: str, 
                     file_path: str) -> None:
        """Store embeddings and metadata in Pinecone."""
        try:
            # Prepare vectors in parallel using list comprehension
            vectors = [
                (
                    f"{candidate_id}_{idx}",
                    embedding,
                    {
                        "text": chunk,
                        "file_path": file_path,
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    }
                )
                for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
            ]
            
            # Clear cache for this candidate
            self._clear_cache_for_candidate(candidate_id)
            
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully stored {len(vectors)} vectors for candidate {candidate_id}")
        except Exception as e:
            logger.error(f"Error storing in Pinecone: {str(e)}")
            raise

    def _clear_cache_for_candidate(self, candidate_id: str) -> None:
        """Clear cache entries for a specific candidate."""
        self._cache = {k: v for k, v in self._cache.items() if not k.startswith(candidate_id)}

    def query(self, query: str, embedder: Embedder, top_k: int = 10, filters: Dict = None) -> Dict:
        """Query the Pinecone index with a search query."""
        try:
            # Check cache first
            cache_key = f"query_{hash(query)}_{top_k}_{str(filters)}"
            if cache_key in self._cache:
                logger.info("Returning cached query results")
                return self._cache[cache_key]
            
            # Get query embedding
            query_vector = embedder.embed(query)
            
            # Prepare filter conditions
            filter_conditions = {}
            if filters:
                filter_conditions = {
                    "file_path": {"$in": filters.get("file_paths", [])},
                    "chunk_index": {"$gte": filters.get("min_chunk", 0)},
                    "total_chunks": {"$lte": filters.get("max_chunks", float('inf'))}
                }
            
            # Query Pinecone with filters
            results = self.index.query(
                vector=query_vector,
                top_k=top_k * 3,  # Fetch more results to account for filtering
                include_metadata=True,
                include_values=False,
                filter=filter_conditions if filter_conditions else None
            )
            
            # Use dictionary comprehension for better performance
            candidate_scores = {
                match['id'].rsplit('_', 1)[0]: {
                    'match': match,
                    'score': float(match['score']),
                    'chunk_index': int(match['id'].split('_')[-1])
                }
                for match in results['matches']
            }
            
            # Sort candidates by their best score and chunk proximity
            sorted_candidates = sorted(
                candidate_scores.values(),
                key=lambda x: (x['score'], -x['chunk_index']),  # Higher score and lower chunk index first
                reverse=True
            )
            
            # Take top_k candidates
            unique_results = [item['match'] for item in sorted_candidates[:top_k]]
            
            # Update results with deduplicated matches
            results['matches'] = unique_results
            
            # Cache the results
            self._cache[cache_key] = results
            
            # Log results
            logger.info(f"Retrieved {len(results['matches'])} unique candidates from Pinecone")
            for i, match in enumerate(results['matches'], 1):
                logger.info(f"\nMatch {i}:")
                logger.info(f"Score: {match['score']:.4f}")
                logger.info(f"ID: {match['id']}")
                logger.info(f"Text: {match['metadata']['text'][:200]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return {'matches': []}

    def get_candidate_chunks(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Fetch all chunks for a specific candidate."""
        try:
            # Check cache first
            cache_key = f"chunks_{candidate_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Create a dummy vector of the correct dimension
            dummy_vector = [0.0] * 1024  # Using standard embedding dimension
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True
            )
            
            # Use list comprehension for better performance
            candidate_chunks = [
                match for match in results.get('matches', [])
                if match['id'].startswith(f"{candidate_id}_")
            ]
            
            if not candidate_chunks:
                logger.warning(f"No chunks found for candidate {candidate_id}")
                return []
            
            # Sort chunks by index
            chunks = sorted(
                candidate_chunks,
                key=lambda x: int(x['id'].split('_')[-1])
            )
            
            # Cache the results
            self._cache[cache_key] = chunks
            
            # Verify chunk sequence
            expected_indices = set(range(len(chunks)))
            actual_indices = {int(chunk['id'].split('_')[-1]) for chunk in chunks}
            missing_indices = expected_indices - actual_indices
            
            if missing_indices:
                logger.warning(f"Missing chunk indices for {candidate_id}: {missing_indices}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error fetching chunks for candidate {candidate_id}: {str(e)}")
            raise

    def get_full_resume_text(self, candidate_id: str) -> str:
        """Get the complete resume text by combining all chunks."""
        try:
            chunks = self.get_candidate_chunks(candidate_id)
            if not chunks:
                return ""
            
            full_text = "\n".join(chunk['metadata']['text'] for chunk in chunks)
            return full_text
            
        except Exception as e:
            logger.error(f"Error getting full resume text for candidate {candidate_id}: {str(e)}")
            raise

# Query Processor
class QueryProcessor:
    """Handles query preprocessing and enhancement."""
    
    def __init__(self):
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self._synonym_cache = {}
        except Exception as e:
            logger.error(f"Error initializing QueryProcessor: {str(e)}")
            # Fallback to empty stopwords if NLTK fails
            self.stop_words = set()
            self._synonym_cache = {}
    
    @lru_cache(maxsize=1000)
    def get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word using WordNet."""
        if word in self._synonym_cache:
            return self._synonym_cache[word]
        
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        
        self._synonym_cache[word] = synonyms
        return synonyms
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess the query by removing stopwords and stemming."""
        try:
            # Tokenize
            tokens = word_tokenize(query.lower())
            
            # Remove stopwords and stem
            processed_tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and token.isalnum()
            ]
            
            return ' '.join(processed_tokens)
        except Exception as e:
            logger.error(f"Error in query preprocessing: {str(e)}")
            # Fallback to simple lowercase and alphanumeric filtering
            return ' '.join(token.lower() for token in query.split() if token.isalnum())
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query using synonyms."""
        try:
            tokens = word_tokenize(query.lower())
            expanded_queries = [query]  # Original query
            
            for token in tokens:
                if token not in self.stop_words and token.isalnum():
                    synonyms = self.get_synonyms(token)
                    for synonym in synonyms:
                        new_query = query.replace(token, synonym)
                        expanded_queries.append(new_query)
            
            return expanded_queries
        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}")
            # Fallback to original query only
            return [query]

# Enhanced Pinecone Manager
class EnhancedPineconeManager(PineconeManager):
    """Enhanced version of PineconeManager with advanced features."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.query_processor = QueryProcessor()
        self._cache_lock = threading.Lock()
        self._cache_ttl = config.CACHE_TTL
        self._last_cache_cleanup = time.time()
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > 300:  # Cleanup every 5 minutes
            with self._cache_lock:
                expired_keys = [
                    k for k, v in self._cache.items()
                    if current_time - v.get('timestamp', 0) > self._cache_ttl
                ]
                for k in expired_keys:
                    del self._cache[k]
                self._last_cache_cleanup = current_time
    
    def _update_cache(self, key: str, value: Any):
        """Update cache with timestamp."""
        with self._cache_lock:
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        self._cleanup_expired_cache()
        with self._cache_lock:
            if key in self._cache:
                cache_entry = self._cache[key]
                if time.time() - cache_entry['timestamp'] <= self._cache_ttl:
                    return cache_entry['value']
                del self._cache[key]
        return None
    
    def batch_query(self, queries: List[str], embedder: Embedder, top_k: int = 10) -> List[Dict]:
        """Process multiple queries in parallel."""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.query, query, embedder, top_k)
                for query in queries
            ]
            return [future.result() for future in futures]
    
    def query(self, query: str, embedder: Embedder, top_k: int = 10, filters: Dict = None) -> Dict:
        """Enhanced query method with preprocessing and expansion."""
        try:
            # Preprocess query
            processed_query = self.query_processor.preprocess_query(query)
            
            # Get expanded queries
            expanded_queries = self.query_processor.expand_query(processed_query)
            
            # Check cache for all queries
            cache_key = f"query_{hash(str(expanded_queries))}_{top_k}_{str(filters)}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Process all queries
            all_results = []
            for expanded_query in expanded_queries:
                query_vector = embedder.embed(expanded_query)
                
                # Prepare filter conditions
                filter_conditions = {}
                if filters:
                    filter_conditions = {
                        "file_path": {"$in": filters.get("file_paths", [])},
                        "chunk_index": {"$gte": filters.get("min_chunk", 0)},
                        "total_chunks": {"$lte": filters.get("max_chunks", float('inf'))}
                    }
                
                # Query Pinecone
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k * 3,
                    include_metadata=True,
                    include_values=False,
                    filter=filter_conditions if filter_conditions else None
                )
                all_results.extend(results['matches'])
            
            # Deduplicate and rerank results
            seen_ids = set()
            unique_results = []
            for result in sorted(all_results, key=lambda x: float(x['score']), reverse=True):
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
                    if len(unique_results) >= top_k:
                        break
            
            final_results = {'matches': unique_results}
            self._update_cache(cache_key, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced query: {str(e)}")
            return {'matches': []}

# Main Indexer Class
class ResumeIndexer:
    """Main class for indexing resumes."""
    
    def __init__(self, config: Config):
        """Initialize the indexer with configuration."""
        self.config = config
        self.embedder = Embedder(config)
        self.pinecone = EnhancedPineconeManager(config)
        self.processor = ResumeProcessor(
            tokenizer=self.embedder.tokenizer,
            config=config
        )

    def process_file(self, filename: str) -> None:
        """Process a single resume file with parallel processing."""
        file_path = os.path.join(self.config.DATA_FOLDER, filename)
        logger.info(f"Processing {filename}")
        
        try:
            # Extract text based on file type
            text = (self.processor.extract_text_from_docx(file_path) 
                   if filename.endswith(".docx") 
                   else self.processor.extract_text_from_pdf(file_path))
            
            if not text.strip():
                logger.warning(f"Skipping empty file: {filename}")
                return
            
            # Create chunks
            chunks = self.processor.chunk_text(text)
            if not chunks:
                logger.warning(f"No chunks created for {filename}")
                return
            
            # Generate candidate_id from filename
            candidate_id = self._generate_candidate_id(filename)
            if not candidate_id:
                logger.warning(f"Skipping file with no valid candidate ID: {filename}")
                return

            # Process chunks in parallel
            def process_chunk_batch(batch_chunks, batch_idx):
                try:
                    embeddings = self.embedder.embed(batch_chunks)
                    self.pinecone.store_vectors(embeddings, batch_chunks, candidate_id, file_path)
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1} of {filename}: {e}")

            # Create batches
            batches = [chunks[i:i + self.config.BATCH_SIZE] 
                      for i in range(0, len(chunks), self.config.BATCH_SIZE)]
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
                futures = [
                    executor.submit(process_chunk_batch, batch, idx)
                    for idx, batch in enumerate(batches)
                ]
                # Wait for all futures to complete
                for future in futures:
                    future.result()
                    
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")

    def _generate_candidate_id(self, filename: str) -> Optional[str]:
        """Generate a clean candidate ID from filename."""
        try:
            base_name = os.path.splitext(filename)[0]
            candidate_id = base_name.replace(" ", "_")
            
            # Remove "fractal" and clean up
            candidate_id = candidate_id.replace("_fractal", "").replace("fractal", "")
            candidate_id = candidate_id.replace("(Fractal)", "").replace("(fractal)", "")
            candidate_id = candidate_id.replace("-", "_")
            
            # Clean up extra underscores
            candidate_id = "_".join(filter(None, candidate_id.split("_")))
            
            return candidate_id if candidate_id else None
            
        except Exception as e:
            logger.error(f"Error generating candidate ID from {filename}: {e}")
            return None

    def reindex_all(self) -> None:
        """Process all resume files in the data directory with parallel processing."""
        try:
            files_to_process = [f for f in os.listdir(self.config.DATA_FOLDER) 
                              if f.endswith(('.docx', '.pdf'))]
            logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=min(len(files_to_process), 4)) as executor:
                list(tqdm(
                    executor.map(self.process_file, files_to_process),
                    total=len(files_to_process),
                    desc="Processing files"
                ))
            
            logger.info("Reindexing completed successfully!")
        except Exception as e:
            logger.error(f"Error during reindexing: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    try:
        # Load configuration
        config = Config()
        
        # Initialize and run indexer
        indexer = ResumeIndexer(config)
        indexer.reindex_all()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 