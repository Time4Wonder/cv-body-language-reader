# %% [markdown]
# # Minimal Machine Learning Demo
# In diesem Notebook generieren wir zwei Daten-Cluster und trennen sie 
# mit einer einfachen logistischen Regression. Alles interaktiv in Neovim!

# %%
# 1. Zelle: Imports und Datengenerierung
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

# Wir generieren zwei ineinandergreifende Halbmonde (Klassen 0 und 1)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

print(f"Daten erfolgreich generiert.")
print(f"Features Shape (X): {X.shape} | Labels Shape (y): {y.shape}")

# %%
# 2. Zelle: Modell trainieren
# Wir fitten eine Standard Logistische Regression
model = LogisticRegression()
model.fit(X, y)

acc = model.score(X, y)
print(f"Modell-Training abgeschlossen.")
print(f"Genauigkeit (Accuracy) auf den Trainingsdaten: {acc * 100:.2f}%")

# %%
# 3. Zelle: Entscheidungsgrenze visualisieren
plt.figure(figsize=(8, 5))

# Ein Grid erstellen, um den Raum einzufärben
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Vorhersagen für das gesamte Grid treffen
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Hintergrundfarben der Klassen zeichnen
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)

# Die echten Datenpunkte oben drüber streuen
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Spectral, s=40)

plt.title(f"Logistic Regression Decision Boundary (Acc: {acc*100:.1f}%)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

# Plot anzeigen
plt.show()
