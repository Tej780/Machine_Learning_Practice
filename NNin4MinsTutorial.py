import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def nonlin(x, deriv=False):
    if deriv==True:
        return x*(1-x) #here x is already sigmoid(x)
    return sigmoid(x)

x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

y = np.array([[0],
             [1],
             [1],
             [0]])

#synapses or weights
syn0 = 2*np.random.random((3,4)) -1 #3 input neurons, 4 output neurons
syn1 = 2*np.random.random((4,1)) -1 #4 input neurons, 1 output neuron

#training
for j in range(60000):
    l0 = x # input
    l1 = nonlin(np.dot(l0,syn0)) #matmul input layer with synapse matrix
                                 #then apply sigmoid to get next layer
                                 #i.e. l1 is activation of layer 1
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y-l2 # output error i.e. cost(?)
    if j % 10000 == 0:
        print ("Error: " + str(np.mean(np.abs(l2_error))))

    #backprop and gradient descent
    l2_delta = l2_error*nonlin(l2,deriv=True) #how much to adjust synapse matrix by.
                                              #

    l1_error = l2_delta.dot(syn1.T) #how much l1 contributed to error in l2 # backprop

    l1_delta = l1_error*nonlin(l1,deriv=True)

    #gradient descent
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("After Training:")
print(l2)
