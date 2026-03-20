"""
Détection avec 2 modèles :
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
        # Vérifier si le fichier existe et n'est pas vide
        if not Path(model_path).exists():
            return None
            
        file_size = Path(model_path).stat().st_size
        if file_size < 1000:  # Fichier trop petit = probablement corrompu
            print(f"    Fichier {model_path} trop petit ({file_size} bytes)")
            return None
            
        # Essayer de charger le modèle
        model = YOLO(model_path)
        print(f"   Modèle chargé: {model_path} ({file_size/1024/1024:.1f} MB)")
        return model
        
    except Exception as e:
        print(f"   Erreur chargement {model_path}: {e}")
        return None


def detect_images_dual():
    """Détection avec 2 modèles"""
    
    print(" DÉTECTION DUAL MODEL")
    print("=" * 60)
    
    # Charger les 2 modèles
    print(" Chargement modèles...")
    
    # Modèle 1 : COCO (voitures, personnes, feux)
    model_coco = load_model_safely(config.MODEL_COCO)
    if model_coco is None:
        print(f" Impossible de charger {config.MODEL_COCO}")
        return
    
    # Modèle 2 : Panneaux (si disponible)
    model_signs = None
    if hasattr(config, 'USE_TRAFFIC_SIGN_MODEL') and config.USE_TRAFFIC_SIGN_MODEL:
        model_signs = load_model_safely(config.MODEL_TRAFFIC_SIGNS)
        if model_signs is None:
            print(f"    Panneaux: modèle non disponible ou corrompu")
            print(f"     Solutions:")
            print(f"     1. Exécuter: python download_speed_model.py")
            print(f"     2. Télécharger manuellement un .pt valide")
            print(f"     3. Utiliser yolov8n.pt en copie")
    else:
        print(f"    Panneaux: désactivé dans config")
    
    print(f"   OCR: {'' if OCR_OK else ''}")
    print(f"   Couleur: {'' if COLOR_OK else ''}")
    print("=" * 60)
    
    # Dossier output
    Path(config.OUTPUT_IMAGES).mkdir(parents=True, exist_ok=True)
    
    # Lister images
    input_dir = Path(config.INPUT_IMAGES)
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not images:
        print(f" Aucune image dans {config.INPUT_IMAGES}")
        return
    
    print(f"\n {len(images)} image(s) à traiter\n")
    
    # Traiter chaque image
    for img_idx, img_path in enumerate(images, 1):
        print(f"[{img_idx}/{len(images)}]  {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   Impossible de charger l'image")
            continue
        
        # ==========================================
        # DÉTECTION 1 : Modèle COCO
        # ==========================================
        try:
            results_coco = model_coco.predict(
                source=img,
                conf=config.CONFIDENCE,
                classes=config.CLASSES_COCO,
                device=config.DEVICE,
                save=False,
                verbose=False
            )
            
            coco_count = len(results_coco[0].boxes) if results_coco[0].boxes is not None else 0
            print(f"   COCO: {coco_count} objets")
            
            # Traiter détections COCO
            if results_coco[0].boxes is not None:
                for box in results_coco[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    name = model_coco.names[cls]
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                    
                    # FEU
                    if cls == 9:
                        
                        if COLOR_OK and crop is not None:
                            color = detect_color(crop)
                            # N'afficher que si couleur détectée (pas unknown)
                            if color != 'unknown':
                                draw_light(img, bbox, color)
                                print(f"     Feu: {color.upper()}")
                    
                
                    
                    # AUTRES
                    else:
                        colors = {
                            'car': (0,255,0), 'person': (255,255,0),
                            'truck': (0,255,0), 'bus': (0,255,0),
                            'motorcycle': (255,0,255)
                        }
                        color = colors.get(name, (255,255,255))
                        
                        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        print(f"    - {name}: {conf*100:.1f}%")
                        
        except Exception as e:
            print(f"   Erreur détection COCO: {e}")
            continue
        
        # ==========================================
        # DÉTECTION 2 : Modèle PANNEAUX
        # ==========================================
        signs_count = 0
        if model_signs:
            try:
                results_signs = model_signs.predict(
                    source=img,
                    conf=config.CONFIDENCE,
                    device=config.DEVICE,
                    save=False,
                    verbose=False
                )
                
                if results_signs[0].boxes is not None:
                    signs_count = len(results_signs[0].boxes)
                    print(f"   Panneaux: {signs_count} détectés")
                    
                    # Traiter panneaux
                    for box in results_signs[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        name = model_signs.names[cls]
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                        
                        # Panneaux de vitesse → OCR
                        is_speed_sign = "Limita" in name
                        
                        # STOP → Affichage simple
                        is_stop = name == "STOP"
                        
                        if is_speed_sign:
                            # Panneau de vitesse → Utiliser OCR
                            speed = None
                            if OCR_OK and crop is not None:
                                speed = read_speed_sign(crop)
                            
                            if speed:
                                draw_speed(img, bbox, speed)
                                print(f"     Limitation: {speed} km/h")
                            # Si OCR échoue, ne rien afficher (ignorer ce panneau)
                        elif is_stop:
                            # STOP détecté par best.pt
                            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
                            cv2.putText(img, "STOP", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                            print(f"     STOP")
                        else:
                            # Autres panneaux ignorés
                            continue
                else:
                    print(f"   Panneaux: 0 détectés")
                    
            except Exception as e:
                print(f"   Erreur détection panneaux: {e}")
        else:
            print(f"    Panneaux: modèle non disponible")
        
        # Sauvegarder
        output_path = Path(config.OUTPUT_IMAGES) / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        
        # Afficher statistiques
        stats = f"   Sauvegardé: {output_path.name}"
        if hasattr(config, 'SHOW_STATS') and config.SHOW_STATS:
            stats += f" | COCO: {coco_count} | Panneaux: {signs_count}"
        print(stats)
        print()
    
    print("=" * 60)
    print(f" Résultats dans: {config.OUTPUT_IMAGES}")
    print("=" * 60)


if __name__ == "__main__":
    detect_images_dual()
