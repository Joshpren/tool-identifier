from random import random, shuffle
import numpy as np



class Test:

    def skalarprodukt(self, liste1, liste2):
        if len(liste1) != len(liste2):
            raise Exception("Listen müssen gleich lang sein")

        ergebnis = 0

        for i in range(len(liste1)):
            ergebnis = ergebnis + (liste1[i] * liste2[i])
        return ergebnis

    def heaviside(self, x):
        if x < 0:
            return 0
        else:
            return 1



weights = [random() for i in range(0,3)]
training = [ [1.0, 1.0, 1.0 ], # Sonne scheint, Freunde haben Zeit
             [1.0, 0.0, 1.0,], # Freunde haben Zeit
             [1.0, 1.0, 0.0 ], # Sonne scheint
             [1.0, 0.0, 0.0 ]] # Gar nix von alledem

targets = [ 1.0, 0.0, 0.0, 0.0 ] # Das letzte war kein Schöner Tag
indexes = [i for i in range(0, len(training))]
shuffle(indexes)
test = Test()
anzahl_epochen = 20
alpha = 0.1
print(weights)
for e in range(0, anzahl_epochen):
    print("Starte Epoche ", e)
    for i in indexes:
        inputs = training[i]
        target = targets[i]

        output = test.heaviside(test.skalarprodukt(inputs, weights))
        delta = target - output

        if (delta != 0):
            print("... Fehler erkannt bei Trainingsbeispiel {}".format(i))

        # Magic - die Delta-Regel für alle Gewichte
        for n in range(0, len(weights)):
            weights[n] = weights[n] + delta * alpha * inputs[n]


print(weights)


