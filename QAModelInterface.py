class QAModelInterface:
    """
    Interface for implementing an QA System
    """

    def get_answer(self, context="",question="", **kwargs) -> dict:
        """
        Take a question and context in string
        return a string of extracted answer from context
        """