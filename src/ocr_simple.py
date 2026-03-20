"""
OCR simple pour lire les limitations de vitesse sur les panneaux
Utilise easyocr pour lire les chiffres
"""

import cv2
import re
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Initialiser le lecteur OCR (une seule fois)
_reader = None

def get_reader():
    """Initialise et retourne le lecteur OCR"""
    global _reader
    if _reader is None and EASYOCR_AVAILABLE:
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


def read_speed_sign(crop_img):
    """
    Lit la vitesse sur un panneau de limitation de vitesse
    
    Args:
        crop_img: Image cropée du panneau (BGR)
    
    Returns:
        int or None: Vitesse détectée (20, 30, 50, etc.) ou None
    """
    if crop_img is None or crop_img.size == 0:
        return None
    
    if not EASYOCR_AVAILABLE:
        return None
    
    try:
        # Prétraitement pour améliorer OCR
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionner si trop petit (augmenter taille)
        h, w = gray.shape
        if h < 100 or w < 100:
            scale = max(200 / h, 200 / w)  # Plus grand pour meilleure OCR
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Augmenter contraste
        gray = cv2.equalizeHist(gray)
        
        # Appliquer seuillage pour avoir noir et blanc net
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Essayer aussi l'inverse
        binary_inv = cv2.bitwise_not(binary)
        
        # Débruitage
        binary = cv2.medianBlur(binary, 3)
        binary_inv = cv2.medianBlur(binary_inv, 3)
        
        # OCR sur les deux versions + original
        reader = get_reader()
        if reader is None:
            return None
        
        results1 = reader.readtext(binary, detail=0, allowlist='0123456789')
        results2 = reader.readtext(binary_inv, detail=0, allowlist='0123456789')
        results3 = reader.readtext(gray, detail=0, allowlist='0123456789')
        
        # Combiner les résultats
        all_text = ' '.join(results1 + results2 + results3)
        
        # Extraire tous les nombres
        numbers = re.findall(r'\d+', all_text)
        
        # Filtrer pour avoir des vitesses valides (10-120)
        valid_speeds = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        
        for num_str in numbers:
            num = int(num_str)
            # Chercher la vitesse valide exacte ou proche
            if num in valid_speeds:
                return num  # Correspondance exacte
            elif 10 <= num <= 120:
                # Trouver la vitesse valide la plus proche
                closest = min(valid_speeds, key=lambda x: abs(x - num))
                if abs(closest - num) <= 5:  # Tolérance réduite à 5
                    return closest
        
        return None
        
    except Exception as e:
        return None


def draw_speed(img, bbox, speed):
    """
    Dessine la limitation de vitesse détectée sur l'image
    
    Args:
        img: Image BGR
        bbox: Boîte englobante [x1, y1, x2, y2]
        speed: Vitesse détectée (entier)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Couleur rouge pour limitation de vitesse
    color = (0, 255, 255)
    
    # Dessiner rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    
    # Texte avec la vitesse
    label = f"{speed} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3
    
    # Taille du texte
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Fond pour le texte
    cv2.rectangle(img, 
                 (x1, y1 - text_height - baseline - 10), 
                 (x1 + text_width + 10, y1), 
                 color, -1)
    
    # Texte en blanc
    cv2.putText(img, label, (x1 + 5, y1 - 5),
               font, font_scale, (255, 255, 255), thickness)

