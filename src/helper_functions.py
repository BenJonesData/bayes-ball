import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import tf_keras as tfk
from tqdm import tqdm
import string
import random
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample


class EpochProgressBar(tfk.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs", 1)
        self.progress_bar = tqdm(
            total=self.epochs, desc="Training Progress", unit="epoch"
        )

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()


def get_train_test(
    data: pd.DataFrame, test_size: int = 2, random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_groups = data.loc[:, "season"].unique()

    train_groups, test_groups = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state
    )

    train_data = data.loc[data["season"].isin(train_groups)]
    test_data = data.loc[data["season"].isin(test_groups)]

    return train_data, test_data


def generate_random_string(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


def bootstrap_mse(y_true, y_pred, n_bootstraps=1000, confidence_level=0.95):
    brier_scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        brier_scores.append(brier_score_loss(y_true[indices], y_pred[indices]))

    lower_bound = np.percentile(brier_scores, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(brier_scores, (1 + confidence_level) / 2 * 100)
    
    return (lower_bound, upper_bound)
