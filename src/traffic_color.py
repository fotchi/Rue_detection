# src/traffic_color.py
"""Traffic light color detection"""
import cv2
import numpy as np

def detect_color(crop):
    """Detect traffic light color: red, yellow, green, or unknown"""
   
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # HSV color ranges
    red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    
    # Count pixels
    r = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    y = cv2.countNonZero(yellow)
    g = cv2.countNonZero(green)
    
    if max(r, y, g) < 50:
        return 'unknown'
    
    return ['red', 'yellow', 'green'][np.argmax([r, y, g])]

def draw_light(img, bbox, color):
    """Draw traffic light with color"""
    x1, y1, x2, y2 = map(int, bbox)
    colors = {'red': (0,0,255), 'yellow': (0,255,255), 'green': (0,255,0)}
    c = colors.get(color, (128,128,128))
    
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
    cv2.putText(img, f"Light: {color.upper()}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
