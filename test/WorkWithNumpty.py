import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm # Für den Fortschrittsbalken

i = np.array([1.0, 0.0, 1.0]) # an erster Position das Bias-Neuron
w = np.array([-0.2, 0.3, 0.3])
o = np.heaviside(i.dot(w), 1.0)
print(o)

data = np.genfromtxt(
    'sonar-mine-rock.csv',
    delimiter=',',
    converters={
        -1: lambda s: 1.0 if s == b'M' else 0.0
    })


fulldata = np.insert(data, 0, 1, axis=1) #mit Bias-Neuron welches an die erst Stelle gesetzt wird

np.random.shuffle(fulldata)#datenset wird gemischt

testdata = fulldata[0:31,:]#Daten die später zum Testen gebraucht werden [10-20% der gesamten Daten]
training = fulldata[31:,:]#Daten zum Trainieren des künstlichen Neuronalen Netzwerkes[80-90% der gesamten Daten]

print(testdata.shape, " sind Testdaten")
print(training.shape, " sind Trainingsdaten")

weights = np.random.rand(61) # 1 für das Bias-Neuron, 60 für Merkmale
epochen = 1000
alpha = 0.001

errors = []  # Wir merken uns den Netzfehler für die Epochen

# Einmal über alle Epochen iterieren...
# dank tqdm kriegen wir einen Fortschrittsbalken angezeigt.
for epoche in tqdm(range(0, epochen)):

    error = 0  # Der Fehler in dieser Epoche zurückgesetzt

    # Wisst ihr noch, warum wir das gemacht haben im
    # ersten Tutorial?
    np.random.shuffle(training)  # Nochmal gut durchmischen

    # Dem Netz jeden Trainingsdatensatz einmal präsentieren...
    for i in range(0, len(training)):

        inputs = training[i, 0:61]  # Die ersten 61 Spalten sind Bias + 60 Merkmale
        target = training[i, -1]  # Die letzte Spalte ist 1 für Metall oder 0 für Fels

        # Feuert das Perzeptron?
        output = np.heaviside(inputs.dot(weights), 1.0)

        # Was wurde erwartet, was geliefert?
        delta = target - output

        # Fehler zählen, falls Fehler da
        # und Gewichte aktualisieren
        if (delta != 0):
            error += 1
            weights += delta * inputs * alpha  # Ja, numpy kann das so

    errors.append(error)

plt.figure(figsize=(12,5))
plt.plot(errors)
_ = plt.title("Trainingsfehler über die Epochen", fontsize=16)
plt.show()

anzahl_fehler = 0

# Wir präsentieren alle Zeilen der Testdaten
for test in testdata:

    # Was kommt zurück, wenn wir die Testdaten mit den
    # oben gelernten Gewichten durch die Perzeptron-
    # Aktivierung jagen?
    o = np.heaviside(test[0:61].dot(weights), 1.)

    # in der letzten Spalte von test steht der erwartete Wert
    delta = test[-1] - o

    # Fehler aufgetreten?
    if delta != 0.0:
        anzahl_fehler += 1

    # Die Güte prozentual ausrechnen.
anzahl_daten = len(testdata)
erkannte_daten = (anzahl_daten - anzahl_fehler) / anzahl_daten

print("Güte im Test: {:0.2f}".format(erkannte_daten))
