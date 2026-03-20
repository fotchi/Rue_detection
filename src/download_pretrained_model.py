"""
Télécharger un modèle pré-entraîné depuis Roboflow
"""

from roboflow import Roboflow

rf = Roboflow(api_key="r7a9uoEHWGrR8xhXySCx")
project = rf.workspace("convert-vkjvx").project("traffic-signs-ohupb")
version = project.version(1)

# Télécharger le modèle au format YOLOv8 (pas dataset)
print(" Téléchargement du modèle pré-entraîné...")
model = version.model
model_path = model.download(model_format="yolov8")

print(f" Modèle téléchargé dans: {model_path}")


