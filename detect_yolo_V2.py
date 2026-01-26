import cv2
from ultralytics import YOLO

# ================= YOLO =================
model = YOLO("yolov8n.pt")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 🔥 clé anti-latence

# ================= PARAMETRES =================
personnes_dans_la_salle = 0
porte_roi = [160, 0, 320, 480]  # x, y, w, h
line_x = porte_roi[2] // 2
prev_position = None

# ================= ZOOM & PAN =================
zoom = 100
pan_x = 0
pan_y = 0

def nothing(x):
    pass

cv2.namedWindow("Comptage YOLO")

cv2.createTrackbar("Zoom %", "Comptage YOLO", 100, 200, nothing)
cv2.createTrackbar("Pan X", "Comptage YOLO", 50, 100, nothing)
cv2.createTrackbar("Pan Y", "Comptage YOLO", 50, 100, nothing)

frame_count = 0
detections_cache = None

# ================= BOUCLE =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # ===== Zoom & Pan =====
    zoom = cv2.getTrackbarPos("Zoom %", "Comptage YOLO")
    pan_x = cv2.getTrackbarPos("Pan X", "Comptage YOLO") - 50
    pan_y = cv2.getTrackbarPos("Pan Y", "Comptage YOLO") - 50

    if zoom < 10:
        zoom = 10

    h, w = frame.shape[:2]
    zh = int(h * 100 / zoom)
    zw = int(w * 100 / zoom)

    cx = w // 2 + pan_x * 5
    cy = h // 2 + pan_y * 5

    x1 = max(0, cx - zw // 2)
    y1 = max(0, cy - zh // 2)
    x2 = min(w, cx + zw // 2)
    y2 = min(h, cy + zh // 2)

    frame_view = frame[y1:y2, x1:x2]
    frame_view = cv2.resize(frame_view, (640, 480))

    # ===== YOLO toutes les 3 frames =====
    if frame_count % 3 == 0:
        results = model.predict(frame_view, conf=0.4, imgsz=416, verbose=False)
        detections_cache = results
    frame_count += 1

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

            cv2.rectangle(frame_view, (x1b, y1b), (x2b, y2b), (0,255,0), 2)
            cv2.circle(frame_view, (cx_box, (y1b+y2b)//2), 4, (0,0,255), -1)

    # ===== Porte & ligne =====
    x, y, w, h = porte_roi
    cv2.rectangle(frame_view, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.line(frame_view, (x+line_x,0), (x+line_x,480), (0,255,255), 2)

    # ===== Comptage =====
    if prev_position is not None and current_position is not None:
        if prev_position < line_x and current_position > line_x:
            personnes_dans_la_salle += 1
        elif prev_position > line_x and current_position < line_x:
            personnes_dans_la_salle -= 1

    prev_position = current_position
    personnes_dans_la_salle = max(0, personnes_dans_la_salle)

    cv2.putText(frame_view, f"Personnes: {personnes_dans_la_salle}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Comptage YOLO", frame_view)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
