import functools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    @functools.cached_property
    def classifier(self):
        """Lazy load the sentiment analysis pipeline with truncation enabled."""
        try:
            # Local import to speed up initial service instantiation
            from transformers import pipeline
            logger.info("Loading sentiment-analysis pipeline...")
            # DistilBERT is used for efficient inference.
            # truncation=True ensures inputs > 512 tokens are handled without error.
            pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )
            # Apply 8-bit dynamic quantization to reduce CPU latency.
            import torch
            pipe.model = torch.quantization.quantize_dynamic(
                pipe.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            return pipe
        except Exception as e:
            logger.error(f"Failed to load LLM pipeline: {e}")
            raise RuntimeError(f"Could not initialize LLM classifier: {e}")

    @functools.cached_property
    def _cached_analyze_sentiment(self):
        """Internal cached function to provide per-instance result caching."""
        @functools.lru_cache(maxsize=128)
        def _analyze(text: str):
            # Accessing self.classifier triggers the lazy loading (if not already loaded)
            # and returns the pipeline object which is then called.
            result = self.classifier(text)
            return result[0]
        return _analyze

    def analyze_sentiment(self, text: str):
        """Analyze text sentiment with per-instance caching and automatic truncation."""
        return self._cached_analyze_sentiment(text)

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
