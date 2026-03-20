import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO
from pathlib import Path

def show_menu():
    """Menu principal"""
    print("\n" + "="*60)
    print(" DÉTECTION AVANCÉE - YOLOv8")
    print("="*60)
    print("1. Détecter IMAGES (simple)")
    print("2. Détecter IMAGES (dual: COCO + Panneaux)  ")
    print("3. Détecter VIDÉOS (simple)")
    print("4. Détecter VIDÉOS (dual: COCO + Panneaux)   NOUVEAU")
    print("5. Détecter WEBCAM")
    print("6. Télécharger modèle panneaux")
    print("7. Tester modèles")
    print("8. Quitter")
    print("="*60)

def download_model_menu():
    """Télécharger modèle"""
    try:
        import download_speed_model
        download_speed_model.download_speed_sign_model()
    except Exception as e:
        print(f" Erreur: {e}")
        print("\nCréez le fichier download_speed_model.py")

def test_models():
    """Tester les modèles"""
    print("\n TEST DES MODÈLES")
    print("=" * 60)
    
    import config
    
    # COCO
    try:
        model = YOLO(config.MODEL_COCO)
        print(f" COCO: {config.MODEL_COCO}")
        print(f"   Classes: {len(model.names)}")
    except Exception as e:
        print(f" COCO: {e}")
    
    # Panneaux
    try:
        if Path(config.MODEL_TRAFFIC_SIGNS).exists():
            model = YOLO(config.MODEL_TRAFFIC_SIGNS)
            print(f" Panneaux: {config.MODEL_TRAFFIC_SIGNS}")
            print(f"   Classes: {list(model.names.values())}")
        else:
            print(f"  Panneaux: non téléchargé")
            print("   Exécuter option 6")
    except Exception as e:
        print(f" Panneaux: {e}")
    
    print("=" * 60)

def main():
    """Boucle principale"""
    
    # Créer dossiers
    for folder in ["input/images", "input/videos", "output/images", 
                   "output/videos", "models", "src"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    while True:
        show_menu()
        choice = input("\n Votre choix (1-8): ").strip()
        
        try:
            if choice == "1":
                from src.detect_image import detect_images
                detect_images()
            
            elif choice == "2":
                from src.detect_image_dual import detect_images_dual
                detect_images_dual()
            
            elif choice == "3":
                from src.detect_video import detect_videos
                detect_videos()
            
            elif choice == "4":
                from src.detect_video_dual import detect_videos_dual
                detect_videos_dual()
            
            elif choice == "5":
                from src.detect_webcam import detect_webcam
                detect_webcam()
            
            elif choice == "6":
                download_model_menu()
            
            elif choice == "7":
                test_models()
            
            elif choice == "8":
                print("\n Au revoir!")
                break
            
            else:
                print(" Choix invalide!")
        
        except KeyboardInterrupt:
            print("\n\n Interrompu par l'utilisateur")
        except Exception as e:
            print(f"\n Erreur: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
