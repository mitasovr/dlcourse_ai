import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = 1.0 * np.sum(np.logical_and(prediction == 1, ground_truth == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = 1.0 * np.sum(np.logical_and(prediction == 0, ground_truth == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = 1.0 * np.sum(np.logical_and(prediction == 1, ground_truth == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = 1.0 * np.sum(np.logical_and(prediction == 0, ground_truth == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2 * precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
