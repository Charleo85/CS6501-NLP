from tqdm import tqdm_notebook as tqdm
from random import shuffle
import pickle

import matplotlib.pyplot as plt

def plot_acc(train_acc, dev_acc):
    fig = plt.figure() 
    plt.plot(train_acc, label='train')
    plt.plot(dev_acc, label='dev')
    plt.legend()
    plt.title("Training Results")
    plt.xlabel('#epoch')
    plt.ylabel('accuracy')

def perceptron(train_set, dev_set):
    dim, _ = train_set[0][0].shape
    theta_0 = np.zeros((dim, 1), dtype=np.float64)
    theta_1 = np.zeros((dim, 1), dtype=np.float64)

    eta = 1
    train_acc = []
    dev_acc = []
    for epoch in tqdm(range(5)):
        print("-------------")
        print("epoch %d"%(epoch))
        # train
        acc = 0
        total = 0
        eta /= 2

        shuffle(train_set)
        for x, y in tqdm(train_set, leave=False):
            y_1 = np.dot(theta_0.T, x)
            y_2 = np.dot(theta_1.T, x)
            y_hat = 0 if y_1 > y_2 else 1
            sign = y_hat-y
            if sign == 0: acc += 1
            else:
                theta_0 = theta_0 + eta*sign*x
                theta_1 = theta_1 - eta*sign*x
            total += 1
        percentage = acc/total*100
        print('training accuracy %f %%'%(percentage))
        train_acc.append(percentage)
        
        # validation
        acc = 0
        total = 0
        for x, y in tqdm(dev_set, leave=False):
            y_1 = np.dot(theta_0.T, x)
            y_2 = np.dot(theta_1.T, x)
            y_hat = 0 if y_1 > y_2 else 1
            sign = y_hat-y
            if sign == 0: acc += 1
            total += 1
        percentage = acc/total*100
        print('validation accuracy %f %%'%(percentage))
        dev_acc.append(percentage)
            
    plot_acc(train_acc, dev_acc)

train_X_Y = pickle.load(open('train_X_Y', mode='rb'))
dev_X_Y = pickle.load(open('dev_X_Y', mode='rb'))

perceptron(train_X_Y, dev_X_Y)