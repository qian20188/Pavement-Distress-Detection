from ultralytics import  YOLO

#load a model
modelpath = r"F:\Yolov9\ultralytics-main\ultralytics-main\mycode\runs\detect\train28\train28\weights\best.pt"

model = YOLO(modelpath)  #load a model

#export the model
model.export(format='onnx')