from preprocess import *
from sklearn import naive_bayes
from helper import get_instance_words,get_all_instance_words
from Semi_EM_NB import Semi_EM_MultinomialNB
from sklearn import metrics
from performance_metrics import get_accuracy
from performance_metrics import get_f_measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



def classify(clf, clf_name, trainX, trainY, testX, testY, trainU):
    print(clf_name)
    print("Fitting....")
    clf.fit(trainX, trainY, trainU)
    print("Done Fitting.....")

    result = clf.predict(testX)
    result = np.array(result, dtype=bool)

    accuracy = get_accuracy(result, testY)
    print("\naccuracy\n :", accuracy)
    # Calculate accuracy only when test results are provided
    #if result.shape == testY.shape:
    #    correct_predictions = np.count_nonzero(result == testY)
    #    print("Accuracy: %.6f\n" % (correct_predictions/result.shape[0]))

    return result


def fun(clf, clf_name, trainX, trainY,trainU, testX, testY):
    print("entered f_measure fun")
    clf.fit(trainX, trainY, trainU)

    result = clf.predict(testX)
    # print("accuracy by Semi_EM_NB: ",get_accuracy(result,testY))
    temp = get_f_measure(result,testY)
    # get_f_measure(clf.predict(testX[:x],testY[:x]),testY[:x])
    # inp = [100,200,300,400,500];

    return temp
def fun1(clf, clf_name, trainX, trainY, testX, testY):
    print("entered f_measure fun1")
    # print(np.count_nonzero(trainY))
    # print(np.count_nonzero(testY))

    clf.fit(trainX, trainY)

    result = clf.predict(testX)
    # print(np.count_nonzero(result))
    # print("\n\n")
    temp = get_f_measure(result,testY)
    # get_f_measure(clf.predict(testX[:x],testY[:x]),testY[:x])
    # inp = [100,200,300,400,500];
    
    return temp

# Args: [], "", ""
def filter(training_data_list, training_data_info, training_data_noninfo, test_data_info, test_data_noninfo, trainU_data):

    # Get Training and Testing Data
    mapping = extract_words_and_add_to_dict(training_data_list)
    reverse_mapping = get_reverse_mapping(mapping)
    training_data0 = get_data([training_data_info], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)

    training_data1 = get_data([training_data_noninfo], mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)

    test_data0 = get_data([test_data_info], mapping)
    test_data1 = get_data([test_data_noninfo], mapping)
    testX = np.append(test_data0, test_data1, axis=0)
    
    testY = np.ones(test_data0.shape[0], dtype=int)
    testY = np.append(testY, np.zeros(test_data1.shape[0], dtype=int))

    trainU = get_data([trainU_data], mapping)
    print(trainU.shape)

    inp = [100,200,300,400,500];
    trainX, trainY = shuffle(trainX,trainY)
    testX, testY = shuffle(testX,testY)
    temp = []
    temp1 = []
    for i in range(5):
        x = int(inp[i]*0.6)
        y= int(inp[i]*0.4)
        test = fun(Semi_EM_MultinomialNB(), "Semi_EM_MultinomialNB", trainX[:inp[i]], trainY[:inp[i]], trainU[:y], testX[:inp[i]], testY[:inp[i]])
        test1 = fun1(naive_bayes.MultinomialNB(), "MultinomialNB", trainX[:inp[i]], trainY[:inp[i]], testX[:inp[i]], testY[:inp[i]])
        print("f_measure  Semi_EM_NB: ",test)
        print("f_measure : NB",test1)
        temp.append(test)
        temp1.append(test1)

    plt.plot(inp,temp,'r',label = 'Semi_EM_NB')
    plt.plot(inp,temp1,'b',label = 'NB')
    plt.xlabel("no of training elements")
    plt.ylabel("f measure")
    # plt.gca().legend('Semi_EM_NB','NB')
    plt.legend()
    plt.show()


    # Writes the words corresponding to the instances in the helper file
    # get_all_instance_words(reverse_mapping,testX,"helper.txt")
    # get_instance_words(reverse_mapping,testX[1])

    predictions = classify(Semi_EM_MultinomialNB(), "Semi_EM_MultinomialNB", trainX, trainY, testX, testY, trainU)
    informative_reviews = testX[predictions]
    return (informative_reviews, mapping, reverse_mapping)
