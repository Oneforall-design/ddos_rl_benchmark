## 🟦 **Cadrage du projet**

### Contexte
Le projet s'inscrit dans le cadre d'une étude sur la détection et la mitigation des attaques DDoS (Distributed Denial of Service) à l'aide de techniques d'apprentissage par renforcement (RL). L'objectif est d'évaluer et de comparer différentes stratégies RL pour protéger un réseau simulé contre des attaques DDoS.

### Objectifs
- Implémenter un environnement de simulation pour les attaques DDoS.
- Développer plusieurs agents RL capables de détecter et de réagir aux attaques.
- Comparer les performances des agents selon des critères définis (taux de détection, temps de réaction, impact sur le réseau).
- Documenter les résultats et proposer des pistes d'amélioration.

### Contraintes
- Utiliser Python et des bibliothèques RL standards (e.g., OpenAI Gym, Stable Baselines).
- Assurer la reproductibilité des expériences.
- Respecter un cadre éthique dans la simulation des attaques.

### Livrables
- Code source complet et documenté.
- Rapport détaillé présentant la méthodologie, les résultats et les analyses.
- Présentation orale synthétisant les points clés du projet.

### Planification
1. Recherche bibliographique et définition de l'environnement (Semaine 1-2)
2. Implémentation des agents RL (Semaine 3-5)
3. Expérimentations et collecte des données (Semaine 6-7)
4. Analyse des résultats et rédaction du rapport (Semaine 8-9)
5. Préparation de la présentation finale (Semaine 10)

## 🟦 Phase 1 — Mise en place du projet

### Création de l’environnement Python
Un environnement virtuel a été créé avec :
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Installation des dépendances
Les dépendances suivantes ont été installées :
```
pip install numpy pandas matplotlib seaborn scikit-learn
pip install gymnasium
pip install stable-baselines3
pip install kagglehub
pip install pyarrow
```

### Structure du projet
Mise en place de l’architecture standard :
src/
    agents/
    envs/
    data/
data/raw/

### Téléchargement du dataset CIC-DDoS2019
Le dataset a été téléchargé automatiquement grâce au script :
python -m src.data.download_cicddos2019

### Test de lecture
Un test dans main.py a permis de confirmer la lecture d’un fichier Parquet :
```
df = pd.read_parquet("data/raw/cicddos2019/UDP-training.parquet")
```

## 🟦 Phase 2 — Prétraitement & représentation des états RL

### 🎯 Objectifs
- Charger et fusionner les fichiers bruts du dataset CIC-DDoS2019.  
- Nettoyer, sélectionner et normaliser les features.  
- Structurer les données sous une forme exploitable pour l'apprentissage par renforcement.

### 🔧 Chargement des données brutes
Le pipeline complet de prétraitement est implémenté dans :
```
src/data/preprocessing.py
```

Le chargement fusionne automatiquement tous les fichiers `.parquet` du dossier :
```
data/raw/cicddos2019/
```

Le dataset complet contient :
- **431 371 lignes**  
- **79 colonnes**

### 🧽 Nettoyage des données
- Suppression des colonnes entièrement vides  
- Remplacement des valeurs manquantes (`NaN`) par **0**  
- Ajout d’une colonne `__source_file__` pour la traçabilité  

### 🧩 Sélection des features
Une liste de features candidates a été définie.  
Sur celles proposées, **8** étaient présentes et utilisées :

- Flow Duration  
- Tot Fwd Pkts  
- Tot Bwd Pkts  
- TotLen Fwd Pkts  
- TotLen Bwd Pkts  
- Flow Byts/s  
- Flow Pkts/s  
- Protocol  

La cible est : **Label**

### 📏 Normalisation & Split

- Standardisation via **StandardScaler()**  
- Découpage train/test : **80% / 20%**, stratifié  
- Résultats :
```
X_train : (345096, 8)
X_test  : (86275, 8)
```

### 💾 Sauvegarde des données prétraitées

Les objets suivants sont générés dans :
```
data/processed/
    X_train.npy
    X_test.npy
    y_train.npy
    y_test.npy
    scaler.pkl
```

### ▶️ Exécution du pipeline
```
python -m src.data.preprocessing
```

-------------------------------------
