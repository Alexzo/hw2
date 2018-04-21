



import matplotlib.pyplot as plt
import csv
import math
import numpy as np



def load_data(csvname):
    data = np.matrix(np.genfromtxt(csvname, delimiter=','))
    x = np.asarray(data[:, 0])
    temp = np.ones((np.size(x), 1))
    X = np.concatenate((temp, x), 1)
    y = np.asarray(data[:, 1])
    return X, y, x


def resu(x, y):
    gp = []
    np2 = []
    for i in range(len(x)):
        f = x[i]
        f.shape = (2, 1)
        gp.append(np.dot(f, f.T))

        np2.append(f * y[i])
    return np.dot(np.linalg.pinv(sum(gp)), sum(np2))


X, y, x = load_data("student_debt.csv")
optimal=resu(X,y)
prediction=[optimal[1]*item+optimal[0] for item in x]
pre=[it[0] for it in prediction]
k=[it[0] for it in x]
y1=[it[0] for it in y]


fig,ax =plt.subplots(1,1,figsize=(6,6))

ax.plot(k,pre)
ax.scatter(k,y1)
plt.show()
