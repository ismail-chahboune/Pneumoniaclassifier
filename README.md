# Pneumonia Detection using Deep Learning (ResNet18)

# Description du Projet :
Ce projet utilise un réseau de neurones convolutionnel (CNN), basé sur **ResNet18 pré-entraîné**, pour classer des radiographies thoraciques en deux catégories :

- **NORMAL**
- **PNEUMONIA**

L’objectif est de construire un modèle capable d’aider à la détection automatique de la pneumonie à partir d’images médicales.

---

# Objectifs du Projet
- Charger et prétraiter les images du dataset **Chest X-Ray**.
- Entraîner un modèle ResNet18 pour une classification binaire.
- Évaluer le modèle via Accuracy, Matrice de Confusion et Classification Report.
- Sauvegarder le modèle entraîné et la courbe de performance.

---

# Approche
1. **Prétraitement des images**
   - Redimensionnement en 224×224  
   - Normalisation  
   - Conversion en tenseurs

2. **Utilisation d’un modèle pré-entraîné**
   - ResNet18 avec poids **ImageNet**
   - Modification de la couche fully connected pour 2 classes

3. **Entraînement**
   - Optimiseur Adam  
   - Cross Entropy Loss  
   - 10 époques d’entraînement

4. **Évaluation**
   - Accuracy sur validation et test
   - Rapport de classification détaillé
   - Sauvegarde d'un graphique : `training_history.png`

---

# Structure du Dataset 
Le projet utilise le dataset officiel *Chest X-Ray Images (Pneumonia)* organisé comme suit :

```
chest_xray/
│── train/
│    ├── NORMAL/
│    └── PNEUMONIA/
│── val/
│    ├── NORMAL/
│    └── PNEUMONIA/
│── test/
     ├── NORMAL/
     └── PNEUMONIA/
```

---

# Résultats
Après entraînement :

- Le modèle obtient une **accuracy élevée sur le jeu de test**.
- Un rapport complet est généré (Precision, Recall, F1-score).
- Un fichier image des courbes d'entraînement est sauvegardé :
  - `training_history.png`
- Le modèle final est sauvegardé :
  - `pneumonia_classifier.pth`

---

# Fichiers Principaux
- `main.py` — Script complet du modèle (chargement dataset, training, évaluation)
- `pneumonia_classifier.pth` — Modèle sauvegardé
- `training_history.png` — Graphiques des pertes & accuracy
- `README.md` — Documentation du projet

---

## 👤 Auteur
Projet réalisé par **Chahboune Ismail**
