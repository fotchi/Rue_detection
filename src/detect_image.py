"""
Détection sur images
Usage: python src/detect_image.py
"""

from ultralytics import YOLO
from pathlib import Path
import config

def detect_images():
    """Détecte les objets dans toutes les images du dossier input/images"""
    
    print(" Démarrage de la détection d'images...")
    
    # Charger le modèle (téléchargement auto si nécessaire)
    model = YOLO(config.MODEL_COCO)
    print(f" Modèle {config.MODEL_COCO} chargé!")
    
    # Créer dossier output
    Path(config.OUTPUT_IMAGES).mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images
    input_dir = Path(config.INPUT_IMAGES)
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not images:
        print(f" Aucune image trouvée dans {config.INPUT_IMAGES}")
        return
    
    print(f" {len(images)} image(s) trouvée(s)")
    
    # Détecter sur chaque image
    for img_path in images:
        print(f"\n Analyse: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=config.CONFIDENCE,
            classes=config.CLASSES_COCO,
            device=config.DEVICE,
            save=False,
            verbose=False
        )
        
        # Sauvegarder résultat
        output_path = Path(config.OUTPUT_IMAGES) / f"detected_{img_path.name}"
        results[0].save(str(output_path))
        
        # Afficher stats
        detections = results[0].boxes
        print(f"   {len(detections)} objet(s) détecté(s)")
        
        # Détail par classe
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            print(f"    - {name}: {conf*100:.1f}%")
    
    print(f"\n Résultats sauvegardés dans: {config.OUTPUT_IMAGES}")

if __name__ == "__main__":
    detect_images()
