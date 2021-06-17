from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

def get_metrics(true_labels, predictions, verbose=False):
    """
    Returns all relevant metrics for a model based on the predictions and true labels.
    Optionally prints all metrics in nice format.
    """
    accuracy = accuracy_score(true_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
    conmat = confusion_matrix(true_labels, predictions)
    sensitivity = conmat[1,1] / sum(conmat[1,:])
    specificity = conmat[0,0] / sum(conmat[0,:])

    if verbose is True:
        print(f'accuracy: {accuracy*100:.4f} % \n')

        print(f'balanced accuracy: {balanced_accuracy*100:.4f} %')
        print(f'sensitivity: {sensitivity}')
        print(f'specificity: {specificity} \n')

        print('confusion matrix: ')
        print(f'{conmat} \n')
        print('[["True Negative", "False Positive"] \n ["False Negative", "True Positive"]] \n')

    return accuracy, balanced_accuracy
