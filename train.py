from ultralytics import YOLO 

model_name= 'model_trial_10' #change this to your model name
version = 'v8'
dataset = 'african_wildlife'
img_size =320

model_path = f"./generated_yamls/{model_name}.yaml"
model = YOLO(model_path)

model.train(data="dataset.yaml",  project="tiny_yolo", name=f"{model_name}_{dataset}", optimizer='SGD',  imgsz=img_size,  epochs=300,  batch=64)