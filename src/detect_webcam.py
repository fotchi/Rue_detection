"""
Détection webcam en temps réel
Usage: python src/detect_webcam.py
Appuyez sur 'q' pour quitter
"""

from ultralytics import YOLO
import cv2
import config

def detect_webcam():
    """Détection en temps réel via webcam"""
    
    print(" Démarrage webcam...")
    print("Appuyez sur 'q' pour quitter")
    
    # Charger le modèle
    model = YOLO(config.MODEL_COCO)
    
    # Ouvrir webcam (0 = webcam par défaut)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Erreur: Impossible d'ouvrir la webcam")
        return
    
    print(" Webcam active!")
    
    while True:
        # Lire frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détection
        results = model.predict(
            source=frame,
            conf=config.CONFIDENCE,
            classes=config.CLASSES_TO_DETECT,
            device=config.DEVICE,
            verbose=False
        )
        
        # Afficher résultat annoté
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(" Webcam fermée")

if __name__ == "__main__":
    detect_webcam()
