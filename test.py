

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u
def load_data(csvname):
    data = np.matrix(np.genfromtxt(csvname, delimiter=','))
    x = np.asarray(data[:,0])
    temp = np.ones((np.size(x),1))
    X = np.concatenate((temp,x),1)
    y = np.asarray(data[:,1])
    y = y/y.max()
    return X,y
def logistic(t):
    return 1.0/(1.0+my_exp(-t))
X,y=load_data("bacteria_data.csv")
def resu(x,y,w):
    gp=[]

    for i in range(len(x)):
        f=x[i]
        f.shape=(2,1)
        gp.append((logistic(np.dot(f.T, w))-y[i][0])*logistic(np.dot(f.T, w))*(1-logistic(np.dot(f.T, w)))*f)
    return 2.0*sum(gp)

w0 = np.array([0,2])
w0.shape = (2,1)


def gradient_descent(X, y, w0, lam):
    w_path = []  # container for weights learned at each iteration
    cost_path = []  # container for associated objective values at each iteration
    w_path.append(w0)
    cost = compute_cost(w0)
    cost_path.append(cost)
    w = w0

    # start gradient descent loop
    max_its = 10
    alpha = 10 ** (-2)
    for k in range(max_its):
        # YOUR CODE GOES HERE - compute gradient
        half= resu(X,y,w)
        half.shape=(2,1)
        grad = np.asarray((half + 2.0 * lam * w)[0])



        # take gradient step
        w = w - alpha * grad




        # update path containers
        w_path.append(w)
        cost = compute_cost(w)
        cost_path.append(cost)


    w_path = np.asarray(w_path)
    w_path.shape = (np.shape(w_path)[0], 2)
    #
    cost_path = np.asarray(cost_path)
    cost_path.shape = (np.size(cost_path), 1)

    return w_path, cost_path

# calculate the cost value for a given input weight w
def compute_cost(w):
    temp = 1 / (1 + my_exp(-np.dot(X, w))) - y
    temp = np.dot(temp.T, temp)
    return temp
lam=10**(-1)
print(2.0*lam*w0)
print(gradient_descent(X,y,w0,lam))

# grad.shape=(2,1)
t=np.dot(X[1].T,w0)

