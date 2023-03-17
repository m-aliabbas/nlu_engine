class ClassifierInterface(object):
    """
    An interace for Classifier Models
    """

    def predict(self,text="",labels=[]) -> dict:
        """
        Get prediction from Bert Model with possible classes names
        """
