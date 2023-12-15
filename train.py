from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch

if __name__ == '__main__':
# Use the model
    results = model.train(data="config.yaml", epochs=20)  # train the model
    results = model.val() #validate
