
#  Détection Vidéo avec 2 Modèles

##  NOUVEAU FICHIER : `src/detect_video_dual.py`

"""
Détection VIDÉO avec 2 modèles :
- Modèle COCO : voitures, personnes, feux
- Modèle Panneaux : limitations de vitesse
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO
from pathlib import Path
import cv2
import config

try:
    from ocr_simple import read_speed_sign, draw_speed  # type: ignore
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    from traffic_color import detect_color, draw_light
    COLOR_OK = True
except ImportError:
    COLOR_OK = False


def load_model_safely(model_path):
    """Charger un modèle avec gestion d'erreur"""
    try:
        if not Path(model_path).exists():
            return None
            
        file_size = Path(model_path).stat().st_size
        if file_size < 1000:
            print(f"    Fichier {model_path} trop petit ({file_size} bytes)")
            return None
            
        model = YOLO(model_path)
        print(f"   Modèle chargé: {model_path} ({file_size/1024/1024:.1f} MB)")
        return model
        
    except Exception as e:
        print(f"   Erreur chargement {model_path}: {e}")
        return None


def process_frame(frame, model_coco, model_signs):
    """
    Traite une frame de vidéo
    
    Args:
        frame: Image de la vidéo
        model_coco: Modèle COCO
        model_signs: Modèle panneaux (peut être None)
    
    Returns:
        frame annotée
    """
    # ==========================================
    # DÉTECTION 1 : Modèle COCO
    # ==========================================
    try:
        results_coco = model_coco.predict(
            source=frame,
            conf=config.CONFIDENCE,
            classes=config.CLASSES_COCO,
            device=config.DEVICE,
            save=False,
            verbose=False
        )
        
        # Traiter détections COCO
        if results_coco[0].boxes is not None:
            for box in results_coco[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                name = model_coco.names[cls]
                
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                # FEU
                if cls == 9:
                    if COLOR_OK and crop is not None:
                        color = detect_color(crop)
                        if color != 'unknown':
                            draw_light(frame, bbox, color)
                
                # STOP (si détecté par COCO)
                elif cls == 11:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(frame, "STOP", (x1, y1-10),
                               cv2.FONT_HERSHEY_BOLD, 1.0, (0,0,255), 3)
                
                # AUTRES (voiture, personne...)
                else:
                    colors = {
                        'car': (0,255,0), 'person': (255,255,0),
                        'truck': (0,255,0), 'bus': (0,255,0),
                        'motorcycle': (255,0,255)
                    }
                    color = colors.get(name, (255,255,255))
                    
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    except Exception as e:
        print(f"   Erreur COCO: {e}")
    
    # ==========================================
    # DÉTECTION 2 : Modèle PANNEAUX
    # ==========================================
    if model_signs:
        try:
            results_signs = model_signs.predict(
                source=frame,
                conf=config.CONFIDENCE,
                device=config.DEVICE,
                save=False,
                verbose=False
            )
            
            if results_signs[0].boxes is not None:
                for box in results_signs[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    name = model_signs.names[cls]
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                    
                    # Panneaux de vitesse
                    is_speed_sign = "Limita" in name
                    is_stop = name == "STOP"
                    
                    if is_speed_sign:
                        # OCR pour lire la vitesse
                        speed = None
                        if OCR_OK and crop is not None:
                            speed = read_speed_sign(crop)
                        
                        if speed:
                            draw_speed(frame, bbox, speed)
                    
                    elif is_stop:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                        cv2.putText(frame, "STOP", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        
        except Exception as e:
            print(f"   Erreur panneaux: {e}")
    
    return frame


def detect_videos_dual():
    """Détection sur vidéos avec 2 modèles"""
    
    print(" DÉTECTION VIDÉO DUAL MODEL")
    print("=" * 60)
    
    # Charger les modèles
    print(" Chargement modèles...")
    
    model_coco = load_model_safely(config.MODEL_COCO)
    if model_coco is None:
        print(f" Impossible de charger {config.MODEL_COCO}")
        return
    
    model_signs = None
    if hasattr(config, 'USE_TRAFFIC_SIGN_MODEL') and config.USE_TRAFFIC_SIGN_MODEL:
        model_signs = load_model_safely(config.MODEL_TRAFFIC_SIGNS)
        if model_signs is None:
            print(f"    Panneaux: non disponible (continuer sans)")
    
    print(f"   OCR: {'' if OCR_OK else ''}")
    print(f"   Couleur: {'' if COLOR_OK else ''}")
    print("=" * 60)
    
    # Créer dossier output
    Path(config.OUTPUT_VIDEOS).mkdir(parents=True, exist_ok=True)
    
    # Lister vidéos
    input_dir = Path(config.INPUT_VIDEOS)
    videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + \
             list(input_dir.glob("*.mov")) + list(input_dir.glob("*.mkv"))
    
    if not videos:
        print(f" Aucune vidéo dans {config.INPUT_VIDEOS}")
        return
    
    print(f"\n {len(videos)} vidéo(s) à traiter\n")
    
    # Traiter chaque vidéo
    for vid_idx, video_path in enumerate(videos, 1):
        print(f"[{vid_idx}/{len(videos)}]  {video_path.name}")
        
        # Ouvrir vidéo
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"   Impossible d'ouvrir la vidéo")
            continue
        
        # Propriétés vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   {width}x{height} @ {fps} FPS | {total_frames} frames")
        
        # Créer VideoWriter pour sauvegarder
        output_path = Path(config.OUTPUT_VIDEOS) / f"detected_{video_path.name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Traiter frame par frame
        frame_count = 0
        processed_count = 0
        
        print(f"   Traitement en cours...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Traiter la frame
            try:
                annotated_frame = process_frame(frame, model_coco, model_signs)
                out.write(annotated_frame)
                processed_count += 1
                
                # Afficher progression tous les 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"     {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            except Exception as e:
                print(f"   Erreur frame {frame_count}: {e}")
                out.write(frame)  # Écrire frame non traitée
        
        # Libérer ressources
        cap.release()
        out.release()
        
        print(f"   Vidéo complète: {processed_count}/{total_frames} frames traitées")
        print(f"   Sauvegardé: {output_path.name}\n")
    
    print("=" * 60)
    print(f" Résultats dans: {config.OUTPUT_VIDEOS}")
    print("=" * 60)


if __name__ == "__main__":
    detect_videos_dual()



