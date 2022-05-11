# NOTE: precision and recall divide by 0 when arrays have no 1's

import numpy as np

def get_accuracy(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    correct_predictions = np.count_nonzero(predictions == labels)
    total_predictions = predictions.shape[0]
    return correct_predictions/total_predictions

def get_precision(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    true_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 1))
    false_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 0))
    return true_pos / (true_pos + false_pos)

def get_recall(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    true_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 1))
    false_neg = np.count_nonzero(np.logical_and(predictions == 0, labels == 1))
    return true_pos / (true_pos + false_neg)

def get_f_measure(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    precision = get_precision(predictions, labels)
    recall = get_recall(predictions, labels)
    return (2 * precision * recall) / (precision + recall)
