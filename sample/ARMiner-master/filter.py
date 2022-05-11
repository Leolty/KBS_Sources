from preprocess import *
from sklearn import naive_bayes
from helper import get_instance_words,get_all_instance_words

def classify(clf, clf_name, trainX, trainY, testX, testY):
    print(clf_name)
    print("Fitting....")
    clf.fit(trainX, trainY)
    print("Done Fitting.....")

    result = clf.predict(testX)
    pred_prob = clf.predict_proba(testX)

    pred_prob = pred_prob[:,1]

    result = np.array(result, dtype=bool)
    # Calculate accuracy only when test results are provided
    if result.shape == testY.shape:
        correct_predictions = np.count_nonzero(result == testY)
        print("Accuracy: %.6f\n" % (correct_predictions/result.shape[0]))

    return result, pred_prob

# Args: [], "", ""
def filter(training_data_list, training_data_info, training_data_noninfo, test_data):

    # Get Training and Testing Data
    mapping = extract_words_and_add_to_dict(training_data_list)
    reverse_mapping = get_reverse_mapping(mapping)
    training_data0 = get_data([training_data_info], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)

    training_data1 = get_data([training_data_noninfo], mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)

    testX = get_data([test_data], mapping)
    testY = np.array([])

    # Writes the words corresponding to the instances in the helper file
    # get_all_instance_words(reverse_mapping,testX,"helper.txt")
    # get_instance_words(reverse_mapping,testX[1])

    predictions, pred_prob = classify(naive_bayes.BernoulliNB(), "BernoulliNB", trainX, trainY, testX, testY)
    print("\nInstances sent to filter: %d" % (testX.shape[0]))
    informative_reviews = testX[predictions]
    pred_prob_array = pred_prob[predictions]
    print("Useful instances: %d\n" % (informative_reviews.shape[0]))
    return (informative_reviews, mapping, reverse_mapping, pred_prob_array)
