import cv2
from ultralytics import YOLO

# ================= YOLO =================
model = YOLO("yolov8n.pt")  # ou yolov8n.onnx

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= PARAMETRES =================
personnes_dans_la_salle = 0

# ROI porte (à ajuster)
porte_roi = (160, 0, 320, 480)  # x, y, w, h
line_x = porte_roi[2] // 2      # ligne verticale au centre de la porte

prev_position = None

# ================= BOUCLE =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    x, y, w, h = porte_roi
    porte = frame[y:y+h, x:x+w]

    # Dessin ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # ================= YOLO INFERENCE =================
    results = model.predict(frame, conf=0.4, verbose=False)

    current_position = None

    if len(results) > 0:
        boxes = results[0].boxes

        best_box = None
        min_dist = 1e9

        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:  # person only
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vérifie intersection avec ROI porte
            if x2 < x or x1 > x+w:
                continue

            cx = (x1 + x2) // 2
            dist = abs(cx - (x + line_x))

            if dist < min_dist:
                min_dist = dist
                best_box = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            current_position = cx - x  # position relative à la porte

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(porte, (current_position, cy - y), 4, (0, 0, 255), -1)

    # ================= LIGNE =================
    cv2.line(porte, (line_x, 0), (line_x, h), (0, 255, 255), 2)

    # ================= COMPTAGE =================
    if prev_position is not None and current_position is not None:
        if prev_position < line_x and current_position > line_x:
            personnes_dans_la_salle += 1
        elif prev_position > line_x and current_position < line_x:
            personnes_dans_la_salle -= 1

    prev_position = current_position

    if personnes_dans_la_salle < 0:
        personnes_dans_la_salle = 0

    # ================= AFFICHAGE =================
    cv2.putText(
        frame,
        f"Personnes dans la salle: {personnes_dans_la_salle}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Comptage YOLO", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
