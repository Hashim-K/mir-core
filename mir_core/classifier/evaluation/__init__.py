# mir_core/classifier/evaluation/__init__.py
"""Classifier evaluation metrics (stubs — not yet implemented).

Will provide: accuracy, macro_f1, confusion_matrix when classifierlab is built.
"""


def accuracy(predictions, targets) -> float:
    """Classification accuracy. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")


def macro_f1(predictions, targets) -> float:
    """Macro-averaged F1 score. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")


def confusion_matrix(predictions, targets):
    """Confusion matrix. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")
