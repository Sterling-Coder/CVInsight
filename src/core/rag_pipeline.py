"""
RAG Pipeline
-----------
A Retrieval-Augmented Generation pipeline for resume search and analysis.
"""

# ----------------------------
# Imports
# ----------------------------
# Standard library imports
import logging
import json
import hashlib
import time
import os
import signal
import sys
import math
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Third-party imports
from together import Together
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# Local imports
from config.prompts import get_summary_prompt, get_analysis_prompt, get_final_analysis_prompt, UNWANTED_SUMMARY_PREFIXES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    MODEL_NAME: str = "lgai/exaone-3-5-32b-instruct"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    DEVICE: Optional[str] = None
    MAX_LENGTH: int = 150
    MIN_LENGTH: int = 30
    NUM_BEAMS: int = 4
    INITIAL_FETCH_K: int = 10
    FINAL_RERANKED_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.6
    CACHE_TTL: int = 3600
    CACHE_SIZE: int = 100
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    RATE_LIMIT_DELAY: float = 1.0
    def __post_init__(self):
        if not self.TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY must be set")
        if self.DEVICE is None:
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RAGPipeline:
    """RAG pipeline for resume search and analysis."""
    def __init__(self, config: RAGConfig, indexer):
        self.config = config
        self.indexer = indexer
        self.pinecone_manager = indexer.pinecone
        self._response_cache = {}
        self._last_cache_cleanup = time.time()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize reranker and Together client."""
        try:
            self.client = Together(api_key=self.config.TOGETHER_API_KEY)
            logger.info(f"Loading reranker model: {self.config.RERANKER_MODEL_NAME}")
            self.reranker = CrossEncoder(
                self.config.RERANKER_MODEL_NAME,
                device=self.config.DEVICE,
                max_length=512
            )
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _call_together_api(self, prompt: str) -> str:
        """Call Together API to generate text."""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    raise RuntimeError(f"Failed to call Together API after {self.config.MAX_RETRIES} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                time.sleep(self.config.RATE_LIMIT_DELAY * (attempt + 1))

    def generate_summary(self, text: str, query: str, score: float) -> str:
        """Generate a summary for the given text using Together API."""
        try:
            prompt = get_summary_prompt(
                query=query,
                text=text
            )

            summary = self._call_together_api(prompt)
            
            # Post-process to remove unwanted prefixes
            for prefix in UNWANTED_SUMMARY_PREFIXES:
                if summary.lower().startswith(prefix.lower()):
                    summary = summary[len(prefix):].strip()

            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise RuntimeError(f"Failed to generate summary: {str(e)}")
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > 300:  # Cleanup every 5 minutes
            expired_keys = [
                k for k, v in self._response_cache.items()
                if current_time - v.get('timestamp', 0) > self.config.CACHE_TTL
            ]
            for k in expired_keys:
                del self._response_cache[k]
            self._last_cache_cleanup = current_time
    
    def _format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the prompt with query and retrieved context."""
        context_text = "\n\n".join([
            f"Candidate ID: {match['id'].split('_')[0]} (Resume {i+1}):\n{match['metadata']['text']}"
            for i, match in enumerate(context)
        ])
        
        return get_analysis_prompt(
            query=query,
            context_text=context_text
        )

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank the initial search results using a cross-encoder."""
        if not results:
            return []

        try:
            sentence_pairs = [[query, match['metadata']['text']] for match in results]
            rerank_scores = self.reranker.predict(sentence_pairs, show_progress_bar=False)
            
            for i, match in enumerate(results):
                raw_score = float(rerank_scores[i])
                percentage_score = (1 / (1 + math.exp(-raw_score / 5.0))) * 100
                match['rerank_score'] = raw_score
                match['percentage_score'] = percentage_score

            return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:self.config.FINAL_RERANKED_K]
        
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results
    
    def _summarize_resume(self, resume_text: str, query: str, score: float) -> str:
        """Generate a professional summary for a single resume."""
        try:
            return self.generate_summary(resume_text, query, score)
        except Exception as e:
            logger.error(f"Error generating summary for resume: {str(e)}")
            return f"Error: Could not generate summary due to {str(e)}"

    def _batch_summarize_resumes(self, candidate_data: List[Dict[str, Any]]) -> List[str]:
        """Generate summaries for multiple resumes in parallel."""
        summaries = []
        max_workers = min(8, len(candidate_data))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._summarize_resume, data['resume_text'], data['query'], data['score'])
                for data in candidate_data
            ]
            for future in futures:
                try:
                    summary = future.result(timeout=120)
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing resume: {str(e)}")
                    summaries.append(f"Error: {str(e)}")
        return summaries

    def _generate_final_analysis(self, query: str, detailed_comparison_text: str) -> str:
        """Generate final analysis comparing all candidates."""
        prompt = get_final_analysis_prompt(
            query=query,
            detailed_comparison_text=detailed_comparison_text
        )

        try:
            return self._call_together_api(prompt).strip()
        except Exception as e:
            logger.error(f"Error generating final analysis: {str(e)}")
            raise RuntimeError(f"Failed to generate final analysis: {str(e)}")

    def search_and_analyze(self, query: str) -> Dict[str, Any]:
        """Search resumes and generate analysis based on query."""
        try:
            # Retrieve initial results
            logger.info(f"Searching for candidates matching: {query}")
            initial_results = self.pinecone_manager.query(
                query=query,
                embedder=self.indexer.embedder,
                top_k=self.config.INITIAL_FETCH_K
            )
            
            if not initial_results or not initial_results.get('matches'):
                return {
                    "status": "no_results",
                    "message": "No matching resumes found",
                    "analysis": None
                }
            
            # Deduplicate candidates
            seen_candidates = set()
            unique_matches = [
                match for match in initial_results['matches']
                if match['id'].split('_')[0] not in seen_candidates
                and not seen_candidates.add(match['id'].split('_')[0])
            ]
            
            logger.info(f"Found {len(unique_matches)} unique candidates")
            
            # Filter by similarity threshold
            filtered_matches = [
                match for match in unique_matches
                if float(match['score']) >= self.config.SIMILARITY_THRESHOLD
            ]
            
            if not filtered_matches:
                return {
                    "status": "low_similarity",
                    "message": "No resumes meet the similarity threshold",
                    "analysis": None
                }
            
            # Rerank results
            logger.info("Reranking candidates")
            reranked_matches = self._rerank_results(query, filtered_matches)
            
            if not reranked_matches:
                return {
                    "status": "no_reranked_results",
                    "message": "No relevant results after reranking",
                    "analysis": None
                }
            
            # Fetch candidate info
            logger.info("Fetching all chunks of the candidates")
            candidate_data_for_summarization = []
            
            def fetch_candidate_info(match):
                candidate_id = match['id'].split('_')[0]
                full_resume_text = self.pinecone_manager.get_full_resume_text(candidate_id)
                percentage_score = match.get('percentage_score', 0.0)
                return candidate_id, full_resume_text, percentage_score
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(fetch_candidate_info, match) for match in reranked_matches]
                for future in futures:
                    candidate_id, full_resume_text, percentage_score = future.result()
                    if full_resume_text:
                        candidate_data_for_summarization.append({
                            'candidate_id': candidate_id,
                            'resume_text': full_resume_text,
                            'query': query,
                            'score': percentage_score
                        })
            
            # Generate summaries
            logger.info("Generating candidate summaries")
            summaries = self._batch_summarize_resumes(candidate_data_for_summarization)
            
            # Create resume summaries list
            resume_summaries = [
                {
                    'candidate_id': data['candidate_id'],
                    'score': data['score'],
                    'summary': summary
                }
                for data, summary in zip(candidate_data_for_summarization, summaries)
                if summary and summary.strip()
            ]
            
            if not resume_summaries:
                return {
                    "status": "no_summaries",
                    "message": "Could not generate summaries for any candidates",
                    "analysis": None
                }
            
            logger.info("Summary generation completed")
            
            # Generate analysis
            logger.info("Generating comparative analysis")
            comparative_analysis_context = [
                {
                    'id': data['candidate_id'],
                    'metadata': {'text': data['resume_text']}
                }
                for data in candidate_data_for_summarization
            ]

            detailed_comparison = self._call_together_api(
                self._format_prompt(query, comparative_analysis_context)
            )
            
            logger.info("Generating final analysis")
            final_analysis = self._generate_final_analysis(query, detailed_comparison)
            
            return {
                "status": "success",
                "query": query,
                "num_results": len(reranked_matches),
                "analysis": final_analysis,
                "matches": resume_summaries
            }
        
        except Exception as e:
            logger.error(f"Error in search_and_analyze: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during search and analysis: {str(e)}",
                "analysis": None
            }
    
    def shutdown(self):
        """Clean up resources before shutting down."""
        logger.info("Shutting down RAG pipeline")
        try:
            if self.config.DEVICE == "cuda":
                torch.cuda.empty_cache()
            self._response_cache.clear()
            
            for attr in ['model', 'tokenizer', 'reranker']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            if self.config.DEVICE == "cuda":
                torch.cuda.empty_cache()

def signal_handler(sig, frame, pipeline):
    """Handle Ctrl+C gracefully."""
    logger.info("Received interrupt signal, shutting down...")
    pipeline.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    # Mock classes for standalone testing
    class MockPinecone:
        def query(self, query, embedder, top_k):
            logger.info("MockPinecone.query called")
            return {'matches': [{'id': 'mock_candidate', 'score': 0.95, 'metadata': {'text': 'Experienced Python developer.'}}]}
        
        def get_full_resume_text(self, candidate_id):
            logger.info(f"MockPinecone.get_full_resume_text for {candidate_id}")
            return "Full text of mock resume."

    class MockIndexer:
        pinecone = MockPinecone()
        embedder = None

    try:
        config = RAGConfig()
        pipeline = RAGPipeline(config, indexer=MockIndexer())
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, pipeline))
        
        # Example usage
        query = "Find candidates with experience in data modeling"
        result = pipeline.search_and_analyze(query)
        
        print(f"\nQuery: {query}")
        print(f"Status: {result['status']}")
        if result['analysis']:
            print(f"Analysis: {result['analysis']}")
        if result.get('matches'):
            print("Matches:")
            for match in result['matches']:
                print(f"  Candidate ID: {match['candidate_id']}, Score: {match['score']:.4f}")
                print(f"  Summary: {match['summary']}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        pipeline.shutdown()