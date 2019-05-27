import pickle


def save_file(name, data):
    f = open('../data/' + name, 'wb')
    pickle.dump(data, f)
    f.close()

def load_pkl(name):
    file = open('../data/' + name, 'rb')
    f = pickle.load(file)
    file.close()
    return f


def count(data):
    true = 0
    false = 0
    for i in range(len(data)):
        if data[i] == 1:
            true += 1
        else:
            false += 1
    print(true)
    print(false)
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in [(i * 0.01+0.3) for i in range(60)]:
        score = f1(y_true=y_true, y_pred=(y_proba > threshold).astype(int))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    print(search_result)
    return best_threshold,best_score


def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = 0
        possible_positives = 0
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                true_positives += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                possible_positives += 1
        #
        # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + true_positives)
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = 0
        predicted_positives = 0
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                true_positives += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                predicted_positives += 1
        # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        if predicted_positives + true_positives==0:
            return 0
        else:
            precision = true_positives / (predicted_positives + true_positives)
            return precision
    precision = precision(y_true, y_pred)
    print("p",precision)
    recall = recall(y_true, y_pred)
    print("r",recall)
    if precision==0 and recall==0:
        return 0
    else:
        return precision,recall,2*((precision*recall)/(precision+recall))