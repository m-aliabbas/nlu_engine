from MobileBertSquadV2 import MobileBertSquadV2
from MobileBertZS import MobileBertZS
from DistilBertCaseSquad import DistilBertCaseSquad
from DistilRobertaBaseZS import DistilRobertaBaseZS
from NLInference import NLI

class NLUEngine(object):
    '''
    A class for performing natural language understanding (NLU) tasks, such as 
    question answering and natural language inference.
    '''

    def __init__(self,**kwargs):
        '''
        Initializes an instance of the NLUEngine class.
        Keyword arguments:
        - qa_engine_type: A string specifying the type of question answering engine to use.
        - qa_engine: A dictionary of keyword arguments to pass to the question answering engine constructor.
        - classifier_type: A string specifying the type of classifier to use.
        - classifier: A dictionary of keyword arguments to pass to the classifier constructor.
        - nli: A dictionary of keyword arguments to pass to the NLI constructor.
        '''
        assert isinstance(kwargs.get("qa_engine_type"), str), "qa_engine_type must be a string"
        assert isinstance(kwargs.get("qa_engine"), dict), "qa_engine must be a dictionary"
        assert isinstance(kwargs.get("classifier_type"), str), "classifier_type must be a string"
        assert isinstance(kwargs.get("classifier"), dict), "classifier must be a dictionary"
        assert isinstance(kwargs.get("nli"), dict), "nli must be a dictionary"
        self.qa_object = self.get_qa_model(kwargs["qa_engine_type"], **kwargs["qa_engine"])
        self.classifier_object = self.get_classifier(kwargs["classifier_type"], **kwargs["classifier"])
        self.nli = NLI(qa_obj=self.qa_object, classifier_obj=self.classifier_object, **kwargs['nli'])

    def get_classifier(self, classifier_type, **kwargs):
        '''
        Returns an instance of a classifier object.
        Arguments:
        - classifier_type: A string specifying the type of classifier to use.
        - kwargs: A dictionary of keyword arguments to pass to the classifier constructor.
        '''
        if classifier_type == "MobileBertZS":
            return MobileBertZS(**kwargs)
        elif classifier_type == "DistilRobertaBaseZS":
            return MobileBertZS(**kwargs)
        else:
            raise ValueError(f"Invalid classifier_type: {classifier_type}")

    def get_qa_model(self, qa_engine_type, **kwargs):
        '''
        Returns an instance of a question answering model object.
        Arguments:
        - qa_engine_type: A string specifying the type of question answering engine to use.
        - kwargs: A dictionary of keyword arguments to pass to the question answering engine constructor.
        '''
        if qa_engine_type == "MobileBertSQ2":
            return MobileBertSquadV2(**kwargs)
        elif qa_engine_type == "DistilBertCaseSquad":
            return DistilBertCaseSquad(**kwargs)
        else:
            raise ValueError(f"Invalid qa_engine_type: {qa_engine_type}")
