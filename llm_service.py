from transformers import pipeline
import functools

class LLMService:
    @functools.cached_property
    def classifier(self):
        # Using a small, efficient model for sentiment analysis
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @functools.lru_cache(maxsize=100)
    def analyze_sentiment(self, text: str):
        result = self.classifier(text)
        return result[0]

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
