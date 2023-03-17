class NLI(object):
    '''
    
    '''
    def __init__(self, qa_obj,classifier_obj,**kwargs) -> None:
        self.qa_obj = qa_obj
        self.classifier_obj = classifier_obj
        

    def classifiy(self,text="",labels=[],**kwargs) -> dict:
        result = self.classifier_obj.predict(text=text,labels=labels)
        print(text,labels,result)
        return {
            'predicted_label':result['labels'][0],
            'confidence':result['scores'][0]
        }
    def ask(self,context="", question="", **kwargs) -> dict:
        result = self.qa_obj.predict(context=context,question=question)
        return {
            'confidence':result['score'],
            'answer':result['answer'],
        }