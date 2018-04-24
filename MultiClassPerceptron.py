import numpy as np
import random
import matplotlib.pyplot as plt

def perceptron_decision(w,x):
    predict = np.argmax(np.array([[np.dot(np.transpose(w[i]), x)] for i in range(10)]))
    return predict

def perceptron_update(w,x,eta,result):
    predict = perceptron_decision(w,x)
    if predict != result:
        w[predict] = w[predict] - eta * x
        w[result] = w[result] + eta * x
        return 0
    else:
        return 1

def num(filename):
    # training:2436 testing:444
    with open(filename, 'r') as f:
        example = f.readlines()
        example_num = len(example) / 33
    return int(example_num)

def read_file(filename,example_num):
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
            inputs.append(training)
            result = f.read(3)
            results.append(int(result))
            num = num + 1
    #with bias
    for i in range(example_num):
        inputs[i].append(1)
    return inputs,results


def main():
    #training
    training_num = num("optdigits-orig_train.txt")
    testing_num = num("optdigits-orig_test.txt")
    training_inputs = np.array(read_file('optdigits-orig_train.txt',training_num)[0])#np.array
    training_results = np.array(read_file('optdigits-orig_train.txt',training_num)[1])
    #weights = np.array([[0.0 for i in range(32 * 32)] for j in range(10)])
    #weights = np.array([[0.0 for i in range(32 * 32+1)] for j in range(10)])
    #weights = np.array([[random.random() for i in range(32 * 32)] for j in range(10)])
    weights = np.array([[random.random() for i in range(32 * 32 + 1)] for j in range(10)])
    epochs = 10
    training_accurancy=[]
    for i in range(epochs):
        eta = 1/(1+i)
        training_correct = 0
        """
        training_order = list(range(training_num))
        random.shuffle(training_order)
        for j in training_order:
            decision = perceptron_update(weights,training_inputs[j],eta,training_results[j])
            training_correct = training_correct + decision
        """
        for j in range(training_num):
            decision = perceptron_update(weights,training_inputs[j],eta,training_results[j])#if classified right,decision=1
            training_correct = training_correct + decision
        training_accurancy.append(training_correct/training_num)
        print("Epoch"+str(i+1)+":"+str(training_accurancy[i]))

    #training curve
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.title('Training Curve')
    plt.plot(range(epochs),training_accurancy)
    plt.savefig('training_curve.png')

    # testing
    testing_inputs = read_file('optdigits-orig_test.txt', testing_num)[0]
    testing_results = read_file('optdigits-orig_test.txt', testing_num)[1]
    testing_correct = 0
    for i in range(testing_num):
        decision = perceptron_decision(weights,testing_inputs[i])
        if decision == testing_results[i]:
            testing_correct = testing_correct+1
    testing_accurancy = testing_correct/testing_num
    print("testing_accurancy:" + str(testing_accurancy))

    #save weights
    for i in range(10):
        file_name = "weights"+str(i)+".txt"
        np.savetxt(file_name, weights[i])

    #confusion martix
    total = [[0 for i in range(10)] for j in range(10)]
    count = [[0 for i in range(10)] for j in range(10)]
    confusion = [[0 for i in range(10)] for j in range(10)]
    for i in range(testing_num):
        c = testing_results[i]
        decision = perceptron_decision(weights, testing_inputs[i])
        count[c][decision] = count[c][decision] + 1
        for j in range(10):
            total[c][j] = total[c][j] + 1
    for x in range(10):
        for y in range(10):
            confusion[x][y] = round(count[x][y]/total[x][y],3)
    print(np.array(confusion))

main()




