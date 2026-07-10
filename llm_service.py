import functools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    @functools.cached_property
    def classifier(self):
        """Lazy load and quantize the sentiment analysis pipeline."""
        try:
            # Local imports to keep application startup fast
            from transformers import pipeline
            import torch

            logger.info("Loading sentiment-analysis pipeline...")
            # DistilBERT is used for efficient inference.
            # truncation=True ensures inputs > 512 tokens are handled without error.
            pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )

            # Apply 8-bit dynamic quantization to Linear layers to reduce latency on CPU
            logger.info("Applying dynamic quantization (8-bit)...")
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
        import torch
        @functools.lru_cache(maxsize=128)
        def _analyze(text: str):
            # Using inference_mode for slightly faster execution and less memory
            with torch.inference_mode():
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
