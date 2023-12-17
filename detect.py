

from ultralytics import YOLO
import cv2



video_index = "3"
video_path = "videos and results/"+"video"+video_index+".mp4"
video_path_out = "videos and results/"+"video_out"+video_index+".mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = "runs/detect/train/weights/best.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.33 #score has to be higher than threshold for openCV to draw a rectangle, this is fully optional

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() +" " +f'{score:.2f}', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                   
    cv2.imshow(video_path_out,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit 
        break
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()