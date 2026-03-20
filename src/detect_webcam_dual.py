import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO
from pathlib import Path
import cv2
import config

try:
    from ocr_simple import read_speed_sign, draw_speed
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    from traffic_color import detect_color, draw_light
    COLOR_OK = True
except ImportError:
    COLOR_OK = False


def load_model_safely(model_path):
    """Charger modèle"""
    try:
        if not Path(model_path).exists():
            return None
        model = YOLO(model_path)
        return model
    except:
        return None


def detect_webcam_dual():
    """Webcam avec dual model"""
    
    print(" WEBCAM DUAL MODEL")
    print("=" * 60)
    print("Appuyez sur 'q' pour quitter")
    print("=" * 60)
    
    # Charger modèles
    print(" Chargement modèles...")
    
    model_coco = load_model_safely(config.MODEL_COCO)
    if model_coco is None:
        print(" Modèle COCO introuvable")
        return
    print(f"   COCO chargé")
    
    model_signs = None
    if hasattr(config, 'USE_TRAFFIC_SIGN_MODEL') and config.USE_TRAFFIC_SIGN_MODEL:
        model_signs = load_model_safely(config.MODEL_TRAFFIC_SIGNS)
        if model_signs:
            print(f"   Panneaux chargé")
        else:
            print(f"    Panneaux non disponible")
    
    print(f"   OCR: {'' if OCR_OK else ''}")
    print(f"   Couleur: {'' if COLOR_OK else ''}")
    print("=" * 60)
    
    # Ouvrir webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Webcam introuvable")
        return
    
    print("\n Webcam active!\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Détection COCO
        try:
            results_coco = model_coco.predict(
                source=frame,
                conf=config.CONFIDENCE,
                classes=config.CLASSES_COCO,
                device=config.DEVICE,
                verbose=False
            )
            
            if results_coco[0].boxes is not None:
                for box in results_coco[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    name = model_coco.names[cls]
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
                    
                    # Feu
                    if cls == 9:
                        if COLOR_OK and crop is not None:
                            color = detect_color(crop)
                            if color != 'unknown':
                                draw_light(frame, bbox, color)
                    
                    # Stop
                    elif cls == 11:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                        cv2.putText(frame, "STOP", (x1,y1-10),
                                   cv2.FONT_HERSHEY_BOLD, 1.0, (0,0,255), 3)
                    
                    # Autres
                    else:
                        colors = {
                            'car': (0,255,0), 'person': (255,255,0),
                            'truck': (0,255,0), 'bus': (0,255,0)
                        }
                        color = colors.get(name, (255,255,255))
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, f"{name}", (x1,y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass
        
        # Détection Panneaux
        if model_signs:
            try:
                results_signs = model_signs.predict(
                    source=frame,
                    conf=config.CONFIDENCE,
                    device=config.DEVICE,
                    verbose=False
                )
                
                if results_signs[0].boxes is not None:
                    for box in results_signs[0].boxes:
                        cls = int(box.cls[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        name = model_signs.names[cls]
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
                        
                        if "Limita" in name:
                            if OCR_OK and crop is not None:
                                speed = read_speed_sign(crop)
                                if speed:
                                    draw_speed(frame, bbox, speed)
                        
                        elif name == "STOP":
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                            cv2.putText(frame, "STOP", (x1,y1-10),
                                       cv2.FONT_HERSHEY_BOLD, 1.0, (0,0,255), 3)
            except:
                pass
        
        # Afficher FPS
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Afficher
        cv2.imshow('YOLOv8 Dual Detection', frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n Webcam fermée")


if __name__ == "__main__":
    detect_webcam_dual()
