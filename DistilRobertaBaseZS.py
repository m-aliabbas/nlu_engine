import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from ClassifierInterface import ClassifierInterface

class DistilRobertaBaseZS(ClassifierInterface):
    def __init__(self,model_path="") -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.classifier = pipeline("zero-shot-classification", model=self.model, 
                                   tokenizer=self.tokenizer)
    def predict(self, text="", labels=[],**kwargs) -> dict:
        result = self.classifier(text, labels)
        return result
