"""
Entraîner un modèle YOLOv8 sur le dataset Traffic-Signs
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import os

# Changer le répertoire de travail à la racine du projet
os.chdir(Path(__file__).parent.parent)

# Chemin vers data.yaml (maintenant relatif à la racine)
data_yaml = "Traffic-Signs-1/data.yaml"

# Créer le dossier models
Path("models").mkdir(exist_ok=True)

# Charger un modèle pré-entraîné YOLOv8
model = YOLO("yolov8s.pt")

# Entraîner
print(" Début de l'entraînement...")
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,            # Réduit de 16 à 8 pour éviter erreur mémoire
    device="cpu",
    project="runs/train",
    name="traffic_signs"
)

# Copier le meilleur modèle
best_model = "runs/train/traffic_signs/weights/best.pt"
if Path(best_model).exists():
    shutil.copy(best_model, "models/traffic_signs.pt")
    print(" Modèle créé: models/traffic_signs.pt")
else:
    print(f" Fichier {best_model} non trouvé")
