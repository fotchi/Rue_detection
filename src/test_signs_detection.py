"""
Test de détection des panneaux pour voir ce qui est détecté
"""

from ultralytics import YOLO
from pathlib import Path
import cv2

# Charger le modèle
model_path = "C:\\Users\\LENOVO\\Desktop\\yolov8-simple\\runs\\detect\\runs\\train\\traffic_signs2\\weights\\best.pt"
model = YOLO(model_path)

print(f" Classes disponibles dans le modèle ({len(model.names)}):")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")

print("\n" + "="*60)

# Tester sur les images
input_dir = Path("input/images")
images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

for img_path in images:
    print(f"\n Test sur: {img_path.name}")
    
    # Tester avec différentes confidences
    for conf in [0.05, 0.1, 0.25]:
        results = model.predict(img_path, conf=conf, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  Conf {conf}: {detections} panneaux détectés")
        
        if detections > 0 and conf == 0.05:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                name = model.names[cls]
                print(f"    - {name}: {confidence*100:.1f}%")

print("\n" + "="*60)
print(" Test terminé")

