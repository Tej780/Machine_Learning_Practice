from sklearn import tree

#height,weight, shoe size
body_measurements = [[181,80,44], [177,70,43], [106, 60, 38], [154, 54, 73],
                     [166,65,40],[190,90,47], [175, 64, 39], [177, 70,40],[159,55,37],
                     [171,75,42],[181,85,43]]

gender = ['m','f','f','f','m','m','m','f','m','f','m']

treeclf = tree.DecisionTreeClassifier()

treeclf.fit(body_measurements,gender)

treeprediciton = treeclf.predict([[190,70,43]])

print("Tree: ",treeprediciton)

from sklearn.neighbors import KNeighborsClassifier

KNclf = KNeighborsClassifier()

KNclf.fit(body_measurements,gender)

KNprediciton = KNclf.predict([[190,70,43]])

print("KNN: ",KNprediciton)

from sklearn.neural_network import MLPClassifier

MLPclf = MLPClassifier()

MLPclf.fit(body_measurements,gender)

MLPprediciton = MLPclf.predict([[190,70,43]])

print("MLP: ",MLPprediciton)
