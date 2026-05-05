# 📈 Système de Prévision de Signaux Temporels

**Signal Forecasting System with Uncertainty Bands**

Système complet pour charger, analyser et prévoir des signaux temporels provenant de fichiers Excel, avec génération d'enveloppes d'incertitude (bandes de confiance).

---

## 🎯 Objectifs

✅ Charger automatiquement tous les signaux individuels des fichiers Excel du dataset  
✅ Traiter chaque colonne (sauf l'année) comme un signal distinct  
✅ Générer des prévisions avec enveloppes d'incertitude à 95%  
✅ Visualiser les prévisions de manière interactive  
✅ Exporter les résultats en CSV  

---

## 📁 Structure du Dataset

Le système gère automatiquement les fichiers Excel avec:
- **Première colonne**: Année (Annee, Year, etc.)
- **Autres colonnes**: Signaux individuels à prévoir
- **Feuilles multiples**: Chaque feuille est traitée indépendamment

### Exemple
```
Fichier: Qualité de l'eau.xlsx
├─ Feuille "ObservationData"
│  ├─ Colonne 1: Annee (1999, 2000, ...)
│  ├─ Colonne 2: Taux de potabilité physico-chimique
│  └─ Colonne 3: Taux de potabilité bactériologique
```

---

## 🚀 Démarrage Rapide

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer l'interface web (Recommandé)
```bash
streamlit run forecast_app.py
```

Puis ouvrez: **http://localhost:8501**

### 3. Ou utiliser la CLI
```bash
# Lister tous les signaux
python forecast_cli.py list

# Lister avec détails
python forecast_cli.py list -v

# Prévoir un signal (avec complément automatique)
python forecast_cli.py forecast "Qualité eau"

# Prévoir avec options
python forecast_cli.py forecast "Qualité eau" -p 10 -o output.csv

# Chercher des signaux
python forecast_cli.py search "Production"
```

---

## 🔧 Architecture du Système

### 1. `core/forecast_loader.py`
**Charge et prépare les données**

```python
from core.forecast_loader import SignalLoader

loader = SignalLoader()
signals = loader.load_all_signals()  # Dict[signal_name] -> DataFrame

# Voir tous les signaux
for signal in loader.list_signals():
    print(signal)

# Charger un signal spécifique
df = loader.get_signal("Ma Qualité de l'eau fournie par l'ONEA - Taux de potabilité physico-chimique")
# Retourne un DataFrame avec colonnes: 'ds' (datetime), 'y' (valeurs)
```

**Classe principale: `SignalLoader`**
- `load_all_signals()`: Charge tous les signaux
- `list_signals()`: Liste tous les signaux disponibles
- `get_signal(name)`: Récupère un signal spécifique
- `get_metadata(name)`: Métadonnées du signal (source, années, points)

---

### 2. `core/forecast_model.py`
**Prévision avec Prophet + enveloppes d'incertitude**

```python
from core.forecast_model import SignalForecaster
from core.forecast_loader import SignalLoader

loader = SignalLoader()
df = loader.get_signal("Mon Signal")

# Créer et entraîner le forecaster
forecaster = SignalForecaster()
forecaster.fit(df, "Mon Signal")

# Générer des prévisions (5 ans)
predictions = forecaster.get_uncertainty_range(periods=5)

# predictions contient:
# - year: Année
# - prediction: Valeur prédite (point estimé)
# - lower_bound: Limite inférieure (95%)
# - upper_bound: Limite supérieure (95%)
# - mid_point: Milieu de l'enveloppe
```

**Classe principale: `SignalForecaster`**
- `fit(df, name)`: Entraîner le modèle
- `predict(periods)`: Générer des prévisions brutes
- `get_prediction_with_bounds(periods)`: Prévisions avec intervalles historiques + futurs
- `get_uncertainty_range(periods)`: Enveloppe d'incertitude pour visualisation
- `forecast_batch(signals_dict, periods)`: Prévoir plusieurs signaux à la fois

---

### 3. `forecast_app.py`
**Interface Streamlit interactive**

**Fonctionnalités:**
- 📊 Sélection de signaux dans une liste déroulante
- 🎨 Graphique interactif avec enveloppe d'incertitude
- 📈 Tableau détaillé des prévisions futures
- 📉 Statistiques (min, max, moyenne)
- 📊 Statistiques d'incertitude
- 💾 Export en CSV

---

## 📊 Comprendre les Enveloppes d'Incertitude

Les prévisions ne sont **jamais une ligne unique** mais une **enveloppe** :

```
Valeur
   |
   |     [Limite Supérieure]
   |    /                  \
   |   /  ENVELOPPE         \
   |  /  D'INCERTITUDE       \
   | / (95%)                  \
   |/____________________________\___
   |\ (Limite Inférieure)       /
   | \                         /
   |__\_______________________/__
       |_____________________|
              Temps

- La zone grisée = zone probable (95% de confiance)
- La ligne orange = meilleure prédiction (médiane)
- L'enveloppe s'élargit = plus d'incertitude dans le futur
```

### Interprétation
- **Enveloppe étroite**: Bonne certitude dans la prévision
- **Enveloppe large**: Incertitude élevée (attention!)
- **Élargissement dans le temps**: Normal, l'incertitude augmente

---

## 💡 Exemples d'Utilisation

### Exemple 1: Interface Web (Recommandé)

```bash
streamlit run forecast_app.py
```

1. Ouvrez http://localhost:8501
2. Sélectionnez un signal dans la barre latérale
3. Ajustez le nombre de périodes à prévoir
4. Visualisez le graphique avec enveloppe d'incertitude
5. Téléchargez les résultats en CSV

### Exemple 2: Script Python

```python
from core.forecast_loader import SignalLoader
from core.forecast_model import SignalForecaster

# Charger les données
loader = SignalLoader()
signals = loader.load_all_signals()

# Sélectionner et prévoir un signal
signal_name = loader.list_signals()[0]  # Premier signal
df = loader.get_signal(signal_name)

# Entraîner et prévoir
forecaster = SignalForecaster()
forecaster.fit(df, signal_name)
predictions = forecaster.get_uncertainty_range(periods=10)

print(predictions)
# Affiche:
#   year  lower_bound  upper_bound  mid_point  prediction
# 0  2024   12345.50    12456.75    12401.12   12401.12
# 1  2025   12450.25    12680.90    12565.57   12565.57
```

### Exemple 3: Prévisions Multiples

```python
from core.forecast_loader import SignalLoader
from core.forecast_model import SignalForecaster

loader = SignalLoader()
all_signals = loader.load_all_signals()

# Créer un sous-ensemble de signaux
selected = {k: v for i, (k, v) in enumerate(all_signals.items()) if i < 5}

# Prévoir en batch
results = SignalForecaster.forecast_batch(selected, periods=5)

for signal_name, predictions in results.items():
    if predictions is not None:
        print(f"\n{signal_name}:")
        print(predictions['future'])
```

---

## 🔍 Détails Techniques

### Algorithme: Prophet (Facebook)

**Pourquoi Prophet?**
- ✅ Excellente gestion des données annuelles
- ✅ Intervalles de confiance automatiques
- ✅ Robuste aux données manquantes
- ✅ Pas besoin de paramétrage complexe
- ✅ Rapide et fiable

**Configuration utilisée:**
```python
Prophet(
    yearly_seasonality=False,    # Pas de saisonnalité annuelle
    daily_seasonality=False,     # Données annuelles
    weekly_seasonality=False,    # Données annuelles
    interval_width=0.95,         # Intervalle de confiance 95%
    seasonality_mode='additive'  # Tendance additive
)
```

### Format des Données

Tous les DataFrames utilisent le format Prophet standard:
```python
DataFrame {
    'ds': datetime,  # Date (première jour de chaque année)
    'y': float,      # Valeur du signal
}
```

---

## 🚦 État du Système

Signaux chargés: **60+**

Répartition par source:
- 14 signaux: Evolution des nombres d'abonnés
- 2 signaux: Qualité de l'eau (ONEA)
- 28 signaux: Production et ventes par région (Production + Vente)
- 17 signaux: Ventes par type de client

---

## ⚙️ Configuration Avancée

### Modifier les paramètres de prévision

Dans `core/forecast_model.py`:

```python
forecaster = SignalForecaster(yearly_seasonality=True)  # Activer saisonnalité
```

### Accroître la fenêtre de confiance

Dans `forecast_model.py`:
```python
interval_width=0.99  # 99% au lieu de 95%
```

---

## 🐛 Dépannage

### "Signal not found"
→ Utiliser `list_signals()` pour voir le nom exact

### "Not enough data points"
→ Le signal a moins de 3 points. Impossible de prévoir.

### Enveloppe trop large
→ Normal pour un signal volatil. Plus de données aide.

### Streamlit ne démarre pas
→ Vérifier: `pip install streamlit prophet pandas plotly`

---

## 📝 Licence

Projet BLUE-Gen - Système de Prévision de Signaux

---

## 👨‍💻 Support

Pour plus d'informations sur Prophet:  
https://facebook.github.io/prophet/

Pour les problèmes Streamlit:  
https://docs.streamlit.io/

---

**Dernière mise à jour:** Mai 2026
