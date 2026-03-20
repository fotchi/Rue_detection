"""
Détection sur vidéos
Usage: python src/detect_video.py
"""

from ultralytics import YOLO
from pathlib import Path
import config

def detect_videos():
    """Détecte les objets dans toutes les vidéos du dossier input/videos"""
    
    print(" Démarrage de la détection vidéo...")
    
    # Charger le modèle
    model = YOLO(config.MODEL)
    print(f" Modèle {config.MODEL} chargé!")
    
    # Créer dossier output
    Path(config.OUTPUT_VIDEOS).mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les vidéos
    input_dir = Path(config.INPUT_VIDEOS)
    videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
    
    if not videos:
        print(f" Aucune vidéo trouvée dans {config.INPUT_VIDEOS}")
        return
    
    print(f" {len(videos)} vidéo(s) trouvée(s)")
    
    # Détecter sur chaque vidéo
    for video_path in videos:
        print(f"\n Analyse: {video_path.name}")
        
        output_path = Path(config.OUTPUT_VIDEOS) / f"detected_{video_path.name}"
        
        # Détection avec affichage progression
        results = model.predict(
            source=str(video_path),
            conf=config.CONFIDENCE,
            classes=config.CLASSES_TO_DETECT,
            device=config.DEVICE,
            save=True,
            project=str(Path(config.OUTPUT_VIDEOS).parent),
            name=Path(config.OUTPUT_VIDEOS).name,
            exist_ok=True,
            stream=True,
            verbose=True
        )
        
        # Compter frames traitées
        frame_count = 0
        for result in results:
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   {frame_count} frames traitées...")
        
        print(f"   Vidéo complète: {frame_count} frames")
    
    print(f"\n Résultats sauvegardés dans: {config.OUTPUT_VIDEOS}")

if __name__ == "__main__":
    detect_videos()
