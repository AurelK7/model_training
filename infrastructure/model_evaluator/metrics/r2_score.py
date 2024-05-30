from application.ports.ports import ModelEvaluatorInterface
from sklearn.metrics import r2_score
import logging
logger = logging.getLogger(__name__)

class MetricR2Score(ModelEvaluatorInterface):
    """r2_score immplementation of ports.ModelEvaluatorInterface."""
    def __init__(self):
        """Instantiate metric"""
        self.metric = r2_score

    def score(self, y_true, y_pred, *args, **kwargs):
        """
        Evaluate model with metrics.
        :param y_true:
        :param y_pred:
        :param args:
        :param kwargs:
        :return: score
        """
        logger.debug("Evaluating model with metrics...")
        score = self.metric(y_true, y_pred, *args, **kwargs)
        logger.debug(f"Score: {score}")
        return score

