import torch
from transformers import AutoModelForQuestionAnswering,  AutoTokenizer, pipeline
from QAModelInterface import QAModelInterface

class DistilBertCaseSquad(QAModelInterface):
    def __init__(self,model_path='') -> None:

        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
        self.expert = pipeline('question-answering', model=self.model, 
                                 tokenizer=self.tokenizer)
        self.QA_INPUT = {
                    'question': '',
                    'context': ''
                }
        

    def predict(self, context="", question="", **kwargs) -> dict:
        QA_input = {
                    'question': question,
                    'context': context
                }
        
        self.QA_INPUT = QA_input
        print(self.QA_INPUT)
        res = self.expert(self.QA_INPUT)
        return res
