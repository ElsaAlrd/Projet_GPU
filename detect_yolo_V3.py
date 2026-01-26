import cv2
from ultralytics import YOLO

# ================= YOLO =================
model = YOLO("yolov8n.pt")
# Décommente si CUDA dispo
# model.to("cuda")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= PARAMETRES =================
personnes_dans_la_salle = 0
porte_roi = [160, 0, 320, 480]
line_x = porte_roi[2] // 2
prev_position = None

# ================= VUE =================
zoom = 1.0
offset_x = 0
offset_y = 0

frame_skip = 3
frame_id = 0
detections_cache = None

# ================= BOUCLE =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # ===== Gestion clavier =====
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('+'):
        zoom *= 1.1
    elif key == ord('-'):
        zoom /= 1.1
    elif key == ord('r'):
        zoom = 1.0
        offset_x = offset_y = 0
    elif key == 81:   # ←
        offset_x -= 20
    elif key == 83:   # →
        offset_x += 20
    elif key == 82:   # ↑
        offset_y -= 20
    elif key == 84:   # ↓
        offset_y += 20

    # ===== Zoom & pan =====
    zw = int(w / zoom)
    zh = int(h / zoom)
    cx = w // 2 + offset_x
    cy = h // 2 + offset_y

    x1 = max(0, cx - zw // 2)
    y1 = max(0, cy - zh // 2)
    x2 = min(w, cx + zw // 2)
    y2 = min(h, cy + zh // 2)

    view = frame[y1:y2, x1:x2]
    view = cv2.resize(view, (640, 480))

    # ===== YOLO toutes les N frames =====
    if frame_id % frame_skip == 0:
        results = model.predict(view, conf=0.4, imgsz=416, verbose=False)
        detections_cache = results
    frame_id += 1

    current_position = None

    if detections_cache:
        boxes = detections_cache[0].boxes

        best_box = None
        min_dist = 1e9

        for box in boxes:
            if int(box.cls[0]) != 0:
                continue

            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])

            if x2b < porte_roi[0] or x1b > porte_roi[0] + porte_roi[2]:
                continue

            cx_box = (x1b + x2b) // 2
            dist = abs(cx_box - (porte_roi[0] + line_x))

            if dist < min_dist:
                min_dist = dist
                best_box = (x1b, y1b, x2b, y2b)

        if best_box:
            x1b, y1b, x2b, y2b = best_box
            cx_box = (x1b + x2b) // 2
            current_position = cx_box - porte_roi[0]

            cv2.rectangle(view, (x1b, y1b), (x2b, y2b), (0,255,0), 2)
            cv2.circle(view, (cx_box, (y1b+y2b)//2), 4, (0,0,255), -1)

    # ===== Porte & ligne =====
    x, y, w_roi, h_roi = porte_roi
    cv2.rectangle(view, (x,y), (x+w_roi,y+h_roi), (255,0,0), 2)
    cv2.line(view, (x+line_x,0), (x+line_x,480), (0,255,255), 2)

    # ===== Comptage =====
    if prev_position is not None and current_position is not None:
        if prev_position < line_x and current_position > line_x:
            personnes_dans_la_salle += 1
        elif prev_position > line_x and current_position < line_x:
            personnes_dans_la_salle -= 1

    prev_position = current_position
    personnes_dans_la_salle = max(0, personnes_dans_la_salle)

    cv2.putText(view, f"Personnes: {personnes_dans_la_salle}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Comptage YOLO", view)

cap.release()
cv2.destroyAllWindows()
