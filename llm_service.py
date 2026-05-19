"""LLM Service for sentiment analysis with async support."""

from transformers import pipeline
import logging
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    """Service for sentiment analysis using DistilBERT model.
    
    Uses a lightweight, efficient model suitable for production environments.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        try:
            # Using a small, efficient model for sentiment analysis
            # DistilBERT is 40% smaller and 60% faster than BERT
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None  # Return all scores
            )
            logger.info("LLM Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM service: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of input text.
        
        Args:
            text: Input text to analyze (max 512 tokens)
            
        Returns:
            Dict with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
            
        Raises:
            ValueError: If text is empty or too long
        """
        if not text or len(text) == 0:
            raise ValueError("Text cannot be empty")
        
        if len(text) > 5000:
            logger.warning("Text exceeds recommended length, truncating to 5000 chars")
            text = text[:5000]
        
        try:
            result = self.classifier(text)
            # Return primary result with highest score
            if isinstance(result, list) and len(result) > 0:
                primary_result = result[0]  # Get top result
                # Normalize output format
                return {
                    "label": primary_result["label"].upper(),
                    "score": float(primary_result["score"])
                }
            else:
                raise ValueError("No sentiment result returned")
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            raise
    
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Async wrapper for sentiment analysis.
        
        Runs the synchronous analysis in a thread pool to avoid blocking.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_sentiment, text)

if __name__ == "__main__":
    import sys
    
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
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
            result = service.analyze_sentiment(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")
        except Exception as e:
            print(f"Error analyzing '{text}': {e}")
