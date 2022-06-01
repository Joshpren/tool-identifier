import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm # Für den Fortschrittsbalken

# 100 Werte zwischen -10 und +10 für unser x
x = np.linspace(-10,10,100)

# Nun Werte für y erzeugen.
# Im Grunde ist dann y = 15x + 25 + Zufall
y = 15 * x + 25.0 + np.random.randn(100) * 25

X = np.insert(x.reshape(100,1), 0, 1, axis=1)


def berechne_verlustfunktion(verlustfunktion):
    w0 = np.linspace(-10, 40, 40) # unser bias zwischen -10 und 40
    w1 = np.linspace(-10, 40, 40) # die steigung auch
    Z = verlust_fuer_grid(verlustfunktion, w0,w1)
    W0, W1 = np.meshgrid(w0, w1) # macht nur koordinaten
    return W0, W1, Z


def verlust_quadrat(target, output):
    # der Doppel-* bedeutet "hoch" in Python,
    # also ** 2 ist "hoch 2" oder "zum Quadrat"
    return (target - output) ** 2


def verlust_fuer_gewichte(verlustfunktion, w0, w1):
    fehler = 0.0

    # Einmal über alle Trainingsbeispiele
    for i in range(len(X)):
        # Ausgabe berechnen mit dem Parametern w0 und w1
        output = X[i].dot(np.array([w0, w1]))

        # Was wäre korrekt gewesen?
        target = y[i]

        # Aufrufen der loss function und addieren zum
        # Gesamtfehler
        fehler += verlustfunktion(target, output)

    return fehler

def verlust_fuer_grid(verlustfunktion, W0s, W1s):
    grid = []

    # Doppelschleife. Das kennt ihr doch?
    for w0 in W0s:
        ingrid = []
        for w1 in W1s:
            err = verlust_fuer_gewichte(verlustfunktion, w0, w1)
            ingrid.append(err)
        grid.append(ingrid)
    return np.array(grid)

def plotte_verlustfunktion(W0s, W1s, Z):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0s, W1s, Z, color='red', alpha=.8)
    ax.plot_wireframe(W0s, W1s, Z, color='y', alpha=0.2)
    ax.view_init(20, 35)
    ax.set_ylabel('', fontsize=16)
    ax.set_xlabel('', fontsize=16)
    ax.grid(True)
    ax.set_title('Verlust / Fehler', fontsize=16)

epochen = 250 # Anzahl der Epochen, jede Epoche = alle Daten zeigen
alpha = 0.0001 # Unsere Lernrate alpha
errors = []   # Nur, damit wir uns die Fehler über die Zeit merken
weights = np.zeros(2) # ein Gewicht für Bias, eines für die Eingabe

# Wir plotten den Urzustand der Linie als schwarze
# gestrichelte Linie erstmal raus
plt.figure(figsize=(12, 9))
plt.scatter(X[:, 1], y, color='r')
plt.plot(X[:, 1], X.dot(weights), color='black', linestyle='--', alpha=.9)

# Wir zippen die X und y Daten in eine Liste zusammen.
# Auf diese Art und Weise können wir einfach data shufflen
# um unsere Daten immer in zufälliger Reihenfolge dem Netz
# zu präsentieren
data = list(zip(X, y))

# Für alle Epochen...
for epoche in tqdm(range(0, epochen)):

    error = 0  # Um die Fehler für jede Epoche zu zählen

    # Einmal durchschütteln
    np.random.shuffle(data)

    # Für alle Datenpunkte...
    for i in range(0, len(X)):
        # Einmal das Netz aktivieren.
        inputs = data[i][0]
        output = inputs.dot(weights)  # ouput = mx + b

        # Was wäre unser Zieldatenpunkt in der Wolke gewesen?
        target = data[i][1]

        # Der Fehler ist unser quadratische Fehler
        error += verlust_quadrat(target, output)

        # Unsere magischer gewordene Delta-Regel
        # Da lineare Aktivierung, keine besondere Ableitung
        # einer Aktivierungsfunktion dazwischen
        weights += alpha * (target - output) * inputs

        # Jetzt haben wir für diese Epoche alle Datenpunkte präsentiert
    # und die Gewichte aktualisiert.

    # Wir merken uns den aufsummierten Fehler der Epoche
    errors.append(error)

    # Wir plotten die aktualisierte Linie in einem Blauton.
    # scale ist eine Hilfsvariable, um eine Skala auf den
    # Blautönen zu finden.
    scale = 1. - ((epochen - epoche) / epochen) / 2.
    plt.plot(X[:, 1], X.dot(weights), color=plt.cm.Blues(scale), alpha=0.8)

# Alle Epochen sind durch, wir plotten noch das finale Ergebnis
# in einem satten Blau
plt.plot(X[:, 1], X.dot(weights), color='b')
plt.grid(True)

# Wir setzen unser gefundenes mx+b in den Titel
_ = plt.title(str(weights[1]) + "x + " + str(weights[0]))
plt.show()