from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
import numpy as np


def find_best_threshold(y_true, y_prob, thresholds):
    """
    Find the best f-score
    """
    
    best_threshold = None
    best_fscore = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        accuracy =  accuracy_score(y_true, y_pred)
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold 
            best_precision = precision
            best_recall = recall
            best_accuracy = accuracy

    
    return best_threshold, best_accuracy, best_recall, best_precision, best_fscore


def scorer(y_true, y_prob):
    """
    Scorer given the y_true and y_prob
    """

    if len(y_true) != len(y_prob):
        raise IndexError('len(y_true) !=  len(y_prob)')
    

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label = 1)
    metric_auc = auc(fpr,tpr)
    
    threshold, acc, recall, precision, fscore = find_best_threshold(y_true, y_prob, thresholds)

    return threshold, thresholds, {'Accuracy': acc, 'Recall': recall, 'Precision':precision, 'Fscore':fscore, 'AUC':metric_auc}


