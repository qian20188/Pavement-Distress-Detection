from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO(r"F:\Yolov9\ultralytics-main\ultralytics-main\yolov8n.pt")  # 加载模型load a pretrained model (recommended for training)
#model = YOLO('helmet.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model.train(data=r"F:\Yolov9\ultralytics-main\ultralytics-main\ultralytics\cfg\datasets\uav.yaml",epochs=100)
# Train the model
results = model.train(data='helmet.yaml', epochs=100, imgsz=640)


