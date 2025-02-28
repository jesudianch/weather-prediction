from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class RainfallPredictor:
    def __init__(self, model):
        """
        Initialize the rainfall predictor with a sklearn-compatible model
        """
        self.model = model
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model using the provided features and target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        """
        Evaluate the model and return performance metrics
        """
        y_pred = self.model.predict(self.X_test)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        return metrics