import functools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    @functools.cached_property
    def classifier(self):
        """Lazy load the sentiment analysis pipeline with quantization and truncation enabled."""
        try:
            # Local imports to speed up initial service instantiation
            import torch
            from transformers import pipeline
            logger.info("Loading sentiment-analysis pipeline...")
            # DistilBERT is used for efficient inference.
            # truncation=True ensures inputs > 512 tokens are handled without error.
            pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )

            # Apply dynamic quantization to the model to speed up CPU inference
            # This converts linear layers to 8-bit integers, reducing latency.
            logger.info("Applying dynamic quantization to the model...")
            quantized_model = torch.quantization.quantize_dynamic(
                pipe.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            # Re-create the pipeline with the quantized model
            return pipeline(
                "sentiment-analysis",
                model=quantized_model,
                tokenizer=pipe.tokenizer,
                truncation=True
            )
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
        """Analyze text sentiment with per-instance caching and automatic truncation.
        Input is normalized to lowercase and stripped to maximize cache hits for uncased model.
        """
        # DistilBERT model used is uncased, so normalizing input maximizes cache hit rate
        # without affecting model accuracy.
        normalized_text = text.lower().strip()
        return self._cached_analyze_sentiment(normalized_text)

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
