from ultralytics import YOLO
import onnx

# Charger le modèle pré-entraîné COCO
model = YOLO("yolov8n.pt")  

# Exporter pour OpenCV DNN
model.export(format="onnx", opset=12, dynamic=True, simplify=True)

model = onnx.load("yolov8n.onnx")
onnx.checker.check_model(model)
print("Modèle ONNX OK")
