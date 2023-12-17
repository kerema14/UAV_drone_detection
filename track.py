import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Open the video file
video_index = "3"
video_path = "videos and results/"+"video"+video_index+".mp4"
video_path_out = "videos and results/"+"video_out"+video_index+".mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame,tracker="bytetrack.yaml", persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()