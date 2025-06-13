import numpy as np 
import os 
import os
import cv2
import sys 
from src.models.forward import Forward
from src.models.backward import Backward
import logging
from tqdm import trange


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Training:
    def __init__(self, models, x_train, y_train, x_test, y_test, epochs=10, batch_size=32, lerning_rate=0.01, save_path='checkpoints'):
        """_summary_

        Args:
            models (_type_): _description_
            x_train (_type_): _description_
            y_train (_type_): _description_
            x_test (_type_): _description_
            y_test (_type_): _description_
            epochs (int, optional): _description_. Defaults to 10.
            batch_size (int, optional): _description_. Defaults to 32.
            lerning_rate (float, optional): _description_. Defaults to 0.01.
            save_path (str, optional): _description_. Defaults to 'checkpoints'.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lerning_rate
        self.save_path = save_path
        
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("x_train anda y_train harus berupa numpy array.")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Jumlah sample pada x_train dan y_train harus sama.")
        if batch_size > x_train.shape[0]:
            raise ValueError("Batch Size tidak boleh lebih besar dari jumlah sample.")
        
        os.makedirs(self.save_path, exist_ok=True)
        
    def _compute_acuracy(self, y_pred, y_true):
        """_summary_

        Args:
            y_pred (_type_): _description_
            y_true (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)
    
    def _save_checkpoint(self, epoch, loss):
        """_summary_

        Args:
            epoch (_type_): _description_
            loss (_type_): _description_
        """
        
        state_path = os.path.join(self.save_path, 'training_state.txt')
        with open(state_path, 'w') as f:
            f.write(f"Epoch: {epoch}, Loss: {loss:.4f}\n")
        
        model_path = os.path.join(self.save_path, f"Model_Epoch_{epoch}.npz")
        np.savez(model_path, **self.models.__dict__)
        logger.info(f"Checkpoint saved at {model_path} for epoch {epoch} with loss {loss:.4f}")
        
    def train(self):
        """
        Melatih model CNN dengan mini-batch gradient descent. 
        """
        logger.info("Memulai Pelatihan CNN...")
        num_samples = self.x_train.shape[0]
        num_batches = num_samples // self.batch_size
        
        start_epoch = 0
        state_path = os.path.join(self.save_path, 'training_state.txt')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = f.read().strip().split(',')
                if len(state) == 2:
                    start_epoch = int(state[0])
                    logger.info(f"Melanjutkan Pelatihan dari epoch")
        try:
            for epoch in trange(start_epoch, self.epochs, desc="Training Epochs", unit="epoch"):
                indices = np.random.permutation(num_samples)
                x_train_shuffled = self.x_train[indices]
                y_train_shuffled = self.y_train[indices]
                
                total_loss = 0.0
                total_accuracy = 0.0
                
                for batch in range(num_batches):
                    start = batch * self.batch_size
                    end = start + self.batch_size
                    x_batch = x_train_shuffled[start:end]
                    y_batch = y_train_shuffled[start:end]
                    
                    outpout = models.forward()

                
                
        except Exception as e:
            logger.info(f"Error During Pelatihan: {str(e)}")
            raise
