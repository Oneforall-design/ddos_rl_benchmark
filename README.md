

# 📘 **Détection d’attaques DDoS par Apprentissage par Renforcement (PPO vs Q-Learning)**

---

## 🔷 1. Introduction

Ce projet vise à comparer l’efficacité de deux algorithmes d’apprentissage par renforcement (RL) – **PPO (Proximal Policy Optimization)** et **Q-Learning** – pour la détection d’attaques DDoS dans un environnement simulé de réseau.

---

## 🔷 2. Objectifs

- Concevoir un environnement simulant des attaques DDoS.
- Implémenter et entraîner des agents RL avec PPO et Q-Learning.
- Comparer leurs performances en termes de détection, précision et temps d’apprentissage.

---

## 🔷 3. Prérequis

- Python 3.8+
- pip

---

## 🔷 4. Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-utilisateur/ddos_rl_benchmark.git
   cd ddos_rl_benchmark
   ```
2. **Créer un environnement virtuel (optionnel mais recommandé) :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```
3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔷 5. Structure du projet

```
ddos_rl_benchmark/
│
├── data/                # Jeux de données simulés ou réels
├── envs/                # Définition de l’environnement RL
├── agents/              # Implémentations PPO et Q-Learning
├── notebooks/           # Analyses et visualisations
├── results/             # Résultats et métriques d’expériences
├── main.py              # Script principal pour lancer les expériences
├── requirements.txt     # Dépendances Python
└── README.md            # Ce fichier
```

---

## 🔷 6. Utilisation

⚠️ *Cette section sera mise à jour lorsque les scripts d’entraînement
(PPO et Q-Learning) seront finalisés.*

Les commandes ci-dessous sont indicatives et seront ajustées :

### Lancer une expérience PPO :
```bash
python main.py --algo ppo --episodes 1000
```

### Lancer une expérience Q-Learning :
```bash
python main.py --algo qlearning --episodes 1000
```

### Options principales :
- `--algo` : Choix de l’algorithme (`ppo` ou `qlearning`)
- `--episodes` : Nombre d’épisodes d’entraînement
- `--render` : Affiche l’environnement (si applicable)

---

## 🔷 7. Résultats attendus

- **Courbes d’apprentissage** : Précision, taux de détection, taux de faux positifs.
- **Comparaison** : Tableaux comparatifs entre PPO et Q-Learning.
- **Reproductibilité** : Scripts et seeds pour répéter les expériences.

---

## 🔷 8. Références

- [OpenAI Gym](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Introduction to Reinforcement Learning (Sutton & Barto)](http://incompleteideas.net/book/the-book.html)

---

## 🔷 9. Auteurs

- **Nathan Hérault** – UQO
- **Bafodé Koulibaly** – UQO

---

## 🔷 10. Licence

Ce projet est sous licence MIT.