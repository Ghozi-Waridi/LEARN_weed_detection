
import numpy as np

class LossFunctions:
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """
        Cross-entropy loss function for multi-class Classification.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]
        return loss
