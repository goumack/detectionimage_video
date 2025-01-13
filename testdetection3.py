import streamlit as st
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np

# Charger le modèle et le processeur
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Vérifier si CUDA est disponible pour utiliser la GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def detect_objects(image):
    """Détecte les objets dans l'image avec optimisation GPU et réduction de la résolution."""
    # Redimensionner l'image pour accélérer le traitement
    image = image.resize((640, 360))  # Réduire la résolution
    
    # Activer le padding lors du prétraitement
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)  # Ajout du padding
    
    # Obtenez les prédictions du modèle
    outputs = model(**inputs)
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Appliquer softmax pour obtenir les probabilités
    probs = torch.nn.functional.softmax(logits[0], dim=-1)
    scores, labels = probs[..., :-1].max(dim=-1)

    # Filtrer les objets avec un seuil de score
    threshold = 0.5
    keep = scores > threshold

    # Ajuster les dimensions des boîtes en fonction de la taille de l'image
    width, height = image.size
    boxes = boxes.squeeze(0)
    
    # Déplacer le tenseur [width, height, width, height] sur le même périphérique que le modèle
    boxes = boxes * torch.tensor([width, height, width, height], device=device)  # Déplacer sur le même périphérique
    
    # Filtrer les boîtes et les scores
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]

    return filtered_boxes, filtered_labels, filtered_scores

def draw_boxes_on_frame(frame, filtered_boxes, filtered_labels, filtered_scores):
    """Dessine les boîtes de détection sur une image OpenCV."""
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        x, y, width, height = map(int, box.tolist())
        class_name = model.config.id2label[label.item()]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    st.title("Détection d'objets en temps réel avec DETR (Webcam)")

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)  # Utilisation de la webcam par défaut

    if not cap.isOpened():
        st.error("Erreur lors de l'ouverture de la webcam.")
        return

    stframe = st.empty()  # Espace réservé pour afficher les cadres vidéo

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Impossible de lire un cadre vidéo.")
            break

        # Convertir le cadre OpenCV (BGR) en PIL (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Détection des objets
        filtered_boxes, filtered_labels, filtered_scores = detect_objects(image)

        # Dessiner les boîtes sur le cadre
        frame = draw_boxes_on_frame(frame, filtered_boxes, filtered_labels, filtered_scores)

        # Afficher le cadre dans Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

if __name__ == "__main__":
    main()
