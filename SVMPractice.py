import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]
])

Y = np.array([-1,-1,1,1,1])

#plot data
for i,x in enumerate(X):
    plt.scatter(x[0],x[1])

#guess LOBF
plt.plot([-2,6],[6,0.5])
plt.show()

def SVM_Plot(X,Y):
    weights = np.zeros(len(X[0]))

    learning_rate = 1

    epochs = 100000

    error_array = []

    for epoch in range(1,epochs+1):
        error = 0
        for i, x in enumerate(X):
            #if miscalssified
            if (Y[i]*np.dot(X[i],weights)<1):
                weights += learning_rate*(X[i]*Y[i] - 2*(1/epoch)*weights)
                error = 1
            else:
                weights += learning_rate*(-2*(1/epoch)*weights)
            error_array.append(error)

    plt.plot(error_array)
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()
    return weights

def ShowSVMFunction(w):
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')

def fit(X,w):
    for i, x in enumerate(X):
        if np.dot(x,w)<0:
            plt.scatter(x[0],x[1],marker='_')
        else:
            plt.scatter(x[0],x[1],marker='+')

w = SVM_Plot(X,Y)

for i,x in enumerate(X):
    if np.dot(x,w)<0:
        plt.scatter(x[0],x[1],marker='_')
    else:
        plt.scatter(x[0],x[1],marker='+')

testData = [[2,2,-1],[4,3,-1]]

fit(testData,w)
ShowSVMFunction(w)
plt.show()
