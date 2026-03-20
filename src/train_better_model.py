"""
Réentraîner le modèle avec plus d'époques pour meilleure précision
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import os

# Changer vers la racine
os.chdir(Path(__file__).parent.parent)

# Charger yolov8s (plus précis que yolov8n)
model = YOLO("yolov8s.pt")

print(" Début de l'entraînement AMÉLIORÉ...")
print("   - Modèle: yolov8s.pt (plus précis)")
print("   - Époques: 100 (plus d'entraînement)")
print("   - Batch: 8")
print("="*60)

# Entraîner avec plus d'époques
results = model.train(
    data="Traffic-Signs-1/data.yaml",
    epochs=100,          # Doublé : 100 au lieu de 50
    imgsz=640,
    batch=8,
    device="cpu",
    patience=20,         # Arrêt automatique si pas d'amélioration
    project="runs/train",
    name="traffic_signs_v2",
    pretrained=True
)

# Copier le meilleur modèle
best_model = "runs/train/traffic_signs_v2/weights/best.pt"
if Path(best_model).exists():
    shutil.copy(best_model, "models/traffic_signs.pt")
    print("\n Nouveau modèle créé: models/traffic_signs.pt")
    print(" Mettez à jour config.py pour pointer vers ce modèle")
else:
    print(f"\n Fichier {best_model} non trouvé")

