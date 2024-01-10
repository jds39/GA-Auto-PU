from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
import numpy as np

class SEM(BaseEstimator, ClassifierMixin):
    def __init__(self, spy_ratio=0.1, tolerance=1e-4, max_iterations=100, base_estimator=None):
        self.spy_ratio = spy_ratio
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.base_estimator = base_estimator if base_estimator is not None else GaussianNB()

    def fit(self, X_train, y_train):
        """ Fit the SEM model to the given training data. """
        # Separate positive and unlabeled samples
        X_positive = X_train[y_train == 1]
        X_unlabeled = X_train[y_train == 0]

        # Initialize spies from the unlabeled data
        X_spies, X_true_unlabeled = self._initialize_spies(X_unlabeled)

        # Label data: Positive (1), Spies (0)
        y_positive = np.ones(len(X_positive))
        y_spies = np.zeros(len(X_spies))

        # Combine positive samples and spies for initial training
        self.X_train_combined = np.vstack((X_positive, X_spies))
        self.y_train_combined = np.concatenate((y_positive, y_spies))

        # Initial model training
        self.base_estimator.fit(self.X_train_combined, self.y_train_combined)

        # Iteratively apply EM steps until convergence
        prev_loss = float('inf')
        for _ in range(self.max_iterations):
            self._expectation_maximization_step(X_true_unlabeled)

            # Calculate the log loss for convergence check
            current_loss = log_loss(self.y_train_combined, self.base_estimator.predict_proba(self.X_train_combined))
            if abs(prev_loss - current_loss) < self.tolerance:
                break
            prev_loss = current_loss

    def _initialize_spies(self, unlabeled_data):
        """ Randomly assign a portion of unlabeled data as spies (negative samples). """
        n_spies = int(len(unlabeled_data) * self.spy_ratio)
        spy_indices = np.random.choice(len(unlabeled_data), n_spies, replace=False)
        spies = unlabeled_data[spy_indices]
        true_unlabeled = np.delete(unlabeled_data, spy_indices, axis=0)
        return spies, true_unlabeled

    def _expectation_maximization_step(self, X_unlabeled):
        """ Perform the E-step and M-step of the EM algorithm. """
        # E-step: Estimate labels for unlabeled data
        estimated_labels = self.base_estimator.predict_proba(X_unlabeled)[:, 1] > 0.5

        # M-step: Retrain the model with estimated labels
        X_combined = np.vstack((self.X_train_combined, X_unlabeled))
        y_combined = np.concatenate((self.y_train_combined, estimated_labels))
        self.base_estimator.fit(X_combined, y_combined)

    def predict(self, X):
        """ Predict the class labels for the provided data. """
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        """ Predict class probabilities for the provided data. """
        return self.base_estimator.predict_proba(X)