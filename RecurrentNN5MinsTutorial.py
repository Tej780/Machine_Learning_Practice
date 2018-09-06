import copy, numpy as np

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidToDerivative(sigmoid):
    return sigmoid * (1 - sigmoid)


intToBinary = {}
maxBinaryLength = 8
maxInt = pow(2, maxBinaryLength)
binary = np.unpackbits(
    np.array([range(maxInt)], dtype=np.uint8).T, axis=1)
for i in range(maxInt):
    intToBinary[i] = binary[i]

learningRate = 0.1
inputDimension = 2
hiddenDimension = 16
outputDimension = 1

weight0 = 2 * np.random.random((inputDimension, hiddenDimension)) - 1
weight1 = 2 * np.random.random((hiddenDimension, outputDimension)) - 1
weightH = 2 * np.random.random((hiddenDimension, hiddenDimension)) - 1  # connects hiddenlayer in
# previous timestep to hidden
# layer in current timestep

weight0update = np.zeros_like(weight0)
weight1update = np.zeros_like(weight1)
weightHupdate = np.zeros_like(weightH)

for j in range(10000):
    intA = np.random.randint((maxInt / 2))
    binA = intToBinary[intA]
    intB = np.random.randint((maxInt / 2))
    binB = intToBinary[intB]

    intC = intA + intB
    binC = intToBinary[intC]

    d = np.zeros_like(binC)

    error = 0

    l2Deltas = []
    l1Values = []
    l1Values.append(np.zeros(hiddenDimension))  # initialise layer 1 values at t=0 as off

    # forward propagate
    for position in range(maxBinaryLength):
        X = np.array([[binA[maxBinaryLength - position - 1], binB[maxBinaryLength - position - 1]]])
        Y = np.array([[binC[maxBinaryLength - position - 1]]]).T

        l1 = sigmoid(np.dot(X, weight0) + np.dot(l1Values[-1], weightH))  # l1 is calculated from
        # both our input and out l1 from previous t
        l2 = sigmoid(np.dot(l1, weight1))  # l2 calculated from l1 as normal

        l2Error = Y - l2
        l2Deltas.append((l2Error) * sigmoidToDerivative(l2))  # l2error*d/dx(sigmoid(x))
        error += np.abs(l2Error[0])

        d[maxBinaryLength - position - 1] = np.round(l2[0][0])  # round output to proper binary value

        l1Values.append(copy.deepcopy(l1))

    future_l1Delta = np.zeros(hiddenDimension)

    # backprop
    for position in range(maxBinaryLength):
        X = np.array([[binA[position], binB[position]]])
        l1 = l1Values[-position - 1]
        prev_l1 = l1Values[-position - 2]

        l2Delta = l2Deltas[-position - 1]
        l1Delta = (future_l1Delta.dot(weightH.T) + l2Delta.dot((weight1.T))) * sigmoidToDerivative(l1)

        weight1update += np.atleast_2d(l1).T.dot(l2Delta)
        weightHupdate += np.atleast_2d(prev_l1).T.dot(l1Delta)
        weight0update += X.T.dot(l1Delta)

        future_l1Delta = l1Delta

    weight0 += weight0update * learningRate
    weight1 += weight1update * learningRate
    weightH += weightHupdate * learningRate

    # empty update variables
    weight0update *= 0
    weight1update *= 0
    weightHupdate *= 0

    if j % 1000 == 0:
        print("Error: ", str(error))
        print("Prediction: ", str(d))
        print("Expected: ", str(binC))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(intA) + " + " + str(intB) + " = " + str(out))
