from typing import AnyStr
from application.ports.ports import ModelEvaluatorInterface
from infrastructure.model_evaluator.metrics.r2_score import MetricR2Score
from infrastructure.model_evaluator.metrics.mae import MetricMeanAbsoluteError
import logging
logger = logging.getLogger(__name__)


class ModelEvaluator(ModelEvaluatorInterface):
    """Base implementation of a port.ModelEvaluatorInterface."""
    def __init__(self, library_name: AnyStr) -> None:
        self.library_name = library_name
        if not isinstance(library_name, list):
            self.library_name = list(library_name)

        available_library = {
            'r2score': MetricR2Score,
            'mae': MetricMeanAbsoluteError
        }
        self.metrics = {}
        for n, library in enumerate(self.library_name):
            if library not in available_library.keys():
                logger.error(f"Library {library} not available in {available_library.keys()}")
                raise Exception(f"Library {library} not available in {available_library.keys()}")
            else:
                self.metrics[library] = available_library[library]()

    def score(self, y_true, y_pred, *args, **kwargs):
        logger.debug(f"Calling evaluator model score with {self.library_name}...")
        scores = {}
        for name, metric in self.metrics.items():
            scores[name] = metric.score(y_true, y_pred, *args, **kwargs)
        return scores

