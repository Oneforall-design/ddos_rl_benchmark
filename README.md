

# Détection d’attaques DDoS par Apprentissage par Renforcement  
**Comparaison PPO vs Q-Learning (DQN) sur CIC-DDoS2019**

Projet réalisé dans le cadre du cours **INF5183 – Fondements de l’intelligence artificielle** (UQO).  
L’objectif est d’étudier la détection d’attaques **DDoS** comme un **problème d’apprentissage par renforcement (RL)**, et de comparer expérimentalement **PPO** et **Q-Learning** (version Deep Q-Network).

---

## 1. Contexte et objectifs

Les attaques par déni de service distribué (**DDoS**) restent une menace majeure pour les infrastructures réseau modernes.  
Les approches classiques de détection reposent surtout sur :

- des règles expertes (IDS basés signatures),
- ou des modèles de classification supervisée statiques.

Dans ce projet, on explore une approche alternative :

> **Formuler la détection DDoS comme une tâche de RL**, où un agent observe des flux réseau et décide s’ils correspondent à du trafic bénin ou à une attaque.

Objectifs principaux :

1. Construire un **environnement Gym** à partir du dataset **CIC-DDoS2019**, adapté au RL.  
2. Implémenter deux agents :
   - un agent **DQN** (Q-Learning approximé par réseau de neurones),
   - un agent **PPO** (Proximal Policy Optimization).  
3. **Comparer** DQN et PPO sur :
   - leurs performances de détection,
   - la stabilité de l’apprentissage,
   - le coût de calcul,
   - la capacité de généralisation.

---

## 2. Jeu de données : CIC-DDoS2019

Le projet utilise **The Canadian Institute for Cybersecurity – DDoS Evaluation Dataset (CIC-DDoS2019)**.

### Téléchargement automatique du dataset

Le dataset peut être téléchargé automatiquement via le script :

```
python src/data/download_cicddos2019.py
```

Ce script récupère les fichiers CSV (ou PCAP selon configuration) et les place dans le dossier `data/raw/`.

---

## 3. Formulation RL

### 3.1 États  
Les états sont des vecteurs continus dérivés des caractéristiques réseau (80+ features).  
Étant donné cette forte dimensionnalité, le Q-Learning tabulaire est impossible, d’où l’utilisation d’un **DQN**.

### 3.2 Actions  
`0 = BENIGN`  
`1 = ATTACK`

### 3.3 Récompenses  
+1 pour une classification correcte,  
−1 sinon, avec possibilité de pénaliser davantage les faux négatifs.

### 3.4 Épisodes  
Les épisodes représentent des séquences temporelles ou des fenêtres glissantes sur le trafic.

---

## 4. Méthodes comparées

### 4.1 DQN  
- Approximation de Q(s,a) via un réseau MLP  
- Replay buffer  
- Target network  
- Politique ε-greedy

### 4.2 PPO  
- Algorithme actor–critic  
- Policy gradient avec clipping  
- Implémentation stable-baselines3

---

## 5. Structure du dépôt

```
ddos_rl_benchmark/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── runs/
├── src/
│   ├── agents/
│   ├── analysis/
│   ├── data/
│   │   ├── cicddos_loader.py
│   │   ├── download_cicddos2019.py
│   │   ├── preprocessing.py
│   │   ├── feature_selection.py
│   │   ├── scaling.py
│   │   └── export_processed.py
│   ├── envs/
│   └── utils/
├── train_dqn.py
├── train_ppo.py
├── eval_dqn.py
├── eval_ppo.py
├── notes.md
├── roadmap.md
└── requirements.txt
```

---

## 6. Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Préparation des données

### Si le dataset n’est pas encore téléchargé :
```
python src/data/download_cicddos2019.py
```

### Prétraitement :
```
python src/data/preprocessing.py
```

---

## 8. Entraînement des agents

DQN :
```
python train_dqn.py
```

PPO :
```
python train_ppo.py
```

---

## 9. Évaluation
```
python eval_dqn.py
python eval_ppo.py
```

Les résultats (matrices de confusion, métriques…) sont sauvegardés dans `figures/` et `runs/`.

---

## 10. Rapport scientifique

Le rapport s’articule autour de :
- description du dataset,
- formulation RL,
- architecture DQN et PPO,
- protocole expérimental,
- comparaison finale.

---

## Licence  
Projet académique — utilisation du dataset CIC-DDoS2019 soumise aux conditions du CIC.