"""
Script pour télécharger un modèle de panneaux depuis Roboflow
Dataset: Traffic-Signs-1 (31 classes de panneaux)
"""

from roboflow import Roboflow

try:
    # Télécharger le dataset
    rf = Roboflow(api_key="r7a9uoEHWGrR8xhXySCx")
    project = rf.workspace("convert-vkjvx").project("traffic-signs-ohupb")
    version = project.version(1)
    dataset = version.download("yolov8")
    
    print(f" Dataset téléchargé: {dataset.location}")
    
except Exception as e:
    print(f" Erreur: {e}")
