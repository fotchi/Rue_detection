"""
Configuration avec modèle de panneaux
"""

from pathlib import Path

# Chemins
INPUT_IMAGES = "input/images"
INPUT_VIDEOS = "input/videos"
OUTPUT_IMAGES = "output/images"
OUTPUT_VIDEOS = "output/videos"

# ==========================================
# MODÈLES
# ==========================================

# Modèle principal (COCO) - pour voitures, personnes, feux
MODEL_COCO = "yolov8n.pt"

# Modèle panneaux - NOUVEAU
MODEL_TRAFFIC_SIGNS = "C:\\Users\\LENOVO\\Desktop\\yolov8-simple\\runs\\detect\\runs\\train\\traffic_signs2\\weights\\best.pt"

# Utiliser quel modèle ?
USE_TRAFFIC_SIGN_MODEL = True  # True = utiliser modèle panneaux

DEVICE = "cpu"
CONFIDENCE = 0.11 # Baissé pour détecter plus de panneaux

# Classes COCO (STOP retiré - géré par best.pt)
CLASSES_COCO = [0, 2, 3, 5, 7, 9]
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
# 9: traffic light

# Affichage
SHOW_LABELS = True
SHOW_CONFIDENCE = True
LINE_THICKNESS = 2

# ==========================================
# Détection couleur feux
# ==========================================
TRAFFIC_LIGHT_COLORS = {
    'red': {
        'lower1': (0, 100, 100),
        'upper1': (10, 255, 255),
        'lower2': (160, 100, 100),
        'upper2': (180, 255, 255)
    },
    'yellow': {
        'lower': (20, 100, 100),
        'upper': (30, 255, 255)
    },
    'green': {
        'lower': (40, 50, 50),
        'upper': (80, 255, 255)
    }
}

MIN_COLOR_AREA = 50
