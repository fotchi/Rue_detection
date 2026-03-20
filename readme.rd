# 🚗 Détection Voitures & Personnes - YOLOv8

Projet simple de détection d'objets sans dataset (modèle pré-entraîné).

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone <votre-repo>
cd yolov8-simple
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/Mac)
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Menu Interactif
```bash
python main.py
```

### Détection sur Images
1. Mettre vos images dans `input/images/`
2. Lancer: `python src/detect_image.py`
3. Résultats dans `output/images/`

### Détection sur Vidéos
1. Mettre vos vidéos dans `input/videos/`
2. Lancer: `python src/detect_video.py`
3. Résultats dans `output/videos/`

### Webcam Temps Réel
```bash
python src/detect_webcam.py
# Appuyer sur 'q' pour quitter
```

## ⚙️ Configuration

Modifier `src/config.py`:
```python
CONFIDENCE = 0.25  # Seuil de confiance
CLASSES_TO_DETECT = [0, 2, 7]  # person, car, truck
```

## 🎯 Classes Détectées

- person (0)
- car (2)
- motorcycle (3)
- bus (5)
- truck (7)
- + 75 autres classes COCO

## 📊 Performance

- CPU: ~50-100 ms/image
- GPU: ~5-10 ms/image
- Modèle: 6 MB (yolov8n)