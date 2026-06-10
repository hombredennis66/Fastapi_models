from functools import cached_property, lru_cache

class LLMService:
    @cached_property
    def classifier(self):
        # Lazy load transformers and the pipeline to improve startup time
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @lru_cache(maxsize=128)
    def analyze_sentiment(self, text: str):
        # Cache results to speed up repeated requests with the same text
        result = self.classifier(text)
        return result[0]

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
