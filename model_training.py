
# model_training.py

from sklearn.metrics import accuracy_score

def compute_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)
