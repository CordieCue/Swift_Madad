import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchreid.models import build_model
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Model Setup ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# YOLO model
yolo_model = YOLO("yolov8n.pt")

# Appearance ReID model (OSNet)
reid_model = build_model(
    name='osnet_x1_0',
    num_classes=0,    # 0 for feature extraction
    pretrained=True
)
reid_model.eval()
reid_model.to(device)

# Transform for ReID
reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Track Database ----------------
track_db = {}  # {track_id: embedding}
next_id = 0
similarity_threshold = 0.76  # cosine similarity threshold

# ---------------- Main Loop ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame, conf=0.4, verbose=False)[0]
    detections = []

    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, classes):
            if cls == 0:  # person class
                detections.append(box)

    # Process each detection
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        # Extract embedding
        x = reid_transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = reid_model(x).cpu().numpy()[0]

        # Match with existing tracks
        assigned = False
        for tid, prev_emb in track_db.items():
            sim = cosine_similarity([embedding], [prev_emb])[0][0]
            if sim > similarity_threshold:
                track_id = tid
                track_db[tid] = embedding  # update embedding
                assigned = True
                break

        if not assigned:
            track_id = next_id
            track_db[next_id] = embedding
            next_id += 1

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO + Appearance ReID Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
