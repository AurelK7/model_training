from application.ports.ports import ModelEvaluatorInterface
from sklearn.metrics import mean_absolute_error
import logging
logger = logging.getLogger(__name__)

class MetricMeanAbsoluteError(ModelEvaluatorInterface):
    """mean_absolute_error metric implementation of ports.ModelEvaluatorInterface."""
    def __init__(self):
        """Instantiate metric"""
        self.metric = mean_absolute_error

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

