"""LLM Service for sentiment analysis with async support and thread pool management."""

from transformers import pipeline
import logging
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LLMService:
    """Service for sentiment analysis using DistilBERT model.
    
    Uses a lightweight, efficient model suitable for production environments.
    Includes proper thread pool management for concurrent async requests.
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize the sentiment analysis pipeline with a bounded thread pool.
        
        Args:
            max_workers: Maximum number of concurrent worker threads (default: 4)
        """
        try:
            # Using DistilBERT because it is 40% smaller and 60% faster than BERT.
            # We omit top_k=None so the pipeline naturally sorts and returns 
            # the highest confidence classification first.
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # PERFORMANCE FIX: Bounding maximum workers prevents the API 
            # from exhausting server memory under heavy concurrent loads.
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            
            logger.info(f"LLM Service initialized successfully with {max_workers} worker threads")
        except Exception as e:
            logger.error(f"Error initializing LLM service: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
            
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        try:
            # SECURITY & ROBUSTNESS FIX: Explicitly passing truncation=True 
            # ensures that if a string hits your Pydantic character limits, 
            # the model tokenizer cuts it safely at 512 tokens instead of throwing an index crash.
            result = self.classifier(text, truncation=True)
            
            if isinstance(result, list) and len(result) > 0:
                primary_result = result[0]  # Safely extracts the top sorted result
                return {
                    "label": primary_result["label"].upper(),
                    "score": round(float(primary_result["score"]), 4)
                }
            else:
                raise ValueError("No sentiment result returned from model pipeline")
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            raise
    
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Async wrapper for sentiment analysis.
        
        Runs the synchronous analysis inside our bounded thread pool to keep the 
        FastAPI event loop completely clear and responsive to other requests.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment analysis results
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.analyze_sentiment, text)

    def shutdown(self) -> None:
        """Cleanly closes down the worker thread pool during application shutdown.
        
        Ensures all pending sentiment analysis tasks complete before shutdown.
        """
        self._executor.shutdown(wait=True)
        logger.info("LLM Service thread pool shut down successfully.")


if __name__ == "__main__":
    # Configure logging for local CLI testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Async test runner to demonstrate the service
    async def test_runner():
        service = LLMService()
        
        test_texts = [
            "I love machine learning!",
            "This is terrible and I hate it.",
            "The weather is okay today."
        ]
        
        print("\nSentiment Analysis Examples:")
        print("=" * 50)
        
        for text in test_texts:
            try:
                result = await service.analyze_sentiment_async(text)
                print(f"\nText: {text}")
                print(f"Sentiment: {result['label']} (confidence: {result['score']})")
            except Exception as e:
                print(f"Error analyzing '{text}': {e}")
        
        service.shutdown()

    asyncio.run(test_runner())
