from numpy import *
from collections import Counter
import numpy as np
import datetime

def num(filename):
    with open(filename, 'r') as f:
        example = f.readlines()
        example_num = len(example) / 33
    return int(example_num)

def read_file(filename,example_num):
        #print(example_num) #training:2436 testing:444
    with open(filename, 'r') as f:
        num = 0
        inputs = []
        results = []
        while num < example_num:
            count = 0
            training = []
            while count < 33 * 32:
                data = f.read(1)
                count = count + 1
                if data != "\n":
                    training.append(float(data))
            # print(training)
            # print(len(training))
            inputs.append(training)
            result = f.read(3)
            results.append(int(result))
            num = num + 1
    return inputs,results

def knn(k):
    training_inputs = np.array(read_file('optdigits-orig_train.txt',training_num)[0])#np.array
    training_results = np.array(read_file('optdigits-orig_train.txt',training_num)[1])
    testing_inputs = read_file('optdigits-orig_test.txt', testing_num)[0]
    testing_results = read_file('optdigits-orig_test.txt', testing_num)[1]
    testing_correct = 0
    total = [[0 for i in range(10)] for j in range(10)]
    count = [[0 for i in range(10)] for j in range(10)]
    confusion = [[0 for i in range(10)] for j in range(10)]

    for i,test in enumerate(testing_inputs):
        #start_time = datetime.datetime.now()
        #cosine

        distances = []
        for train in training_inputs:
            dist = 1 - np.dot(test, train) / (np.linalg.norm(test) * np.linalg.norm(train))
            distances.append(dist)
        distances = np.transpose(np.array(distances))

        #Euclidean Distance
        #distances = np.sqrt(np.square(tile(test,(training_num,1))-training_inputs).sum(axis=1))

        index = distances.argsort()
        classlist = [training_results[index[i]] for i in range(k)]
        test_predict = Counter(classlist).most_common(1)[0][0]
        if test_predict == testing_results[i]:
            testing_correct = testing_correct + 1
        count[testing_results[i]][test_predict] += 1
        for j in range(10):
            total[testing_results[i]][j] += 1
        #end_time = datetime.datetime.now()
        #run_time = end_time - start_time
        #print(run_time)

    testing_accurancy = testing_correct / testing_num
    print("testing_accurancy:" + str(testing_accurancy))
    for x in range(10):
        for y in range(10):
            confusion[x][y] = round(count[x][y] / total[x][y], 3)
    print(np.array(confusion))

training_num = num("optdigits-orig_train.txt")
testing_num = num("optdigits-orig_test.txt")
knn(3)
"""
for i in range(25):
    print("k="+str(i+1))
    knn(i+1)
"""




