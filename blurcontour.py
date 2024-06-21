import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load YOLO model
model = YOLO("yolov8n-seg.pt")  # Use the segmentation model

# Get the class index for 'person'
person_class_index = None
for idx, name in model.names.items():
    if name == 'person':
        person_class_index = idx
        break

cap = cv2.VideoCapture("video1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Blur ratio
blur_ratio = 50

# Video writer
video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Loop to process each frame
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
    clss = results[0].boxes.cls.cpu().tolist()

    if masks is not None:
        for mask, cls in zip(masks, clss):
            if int(cls) == person_class_index:  # Check if the class is 'person'
                # Create a mask for the blur region
                blur_mask = mask.astype(np.uint8) * 255

                # Find the bounding box of the mask
                x, y, w, h = cv2.boundingRect(blur_mask)

                # Create a subregion for blurring
                subregion = im0[y:y+h, x:x+w]

                # Create a mask for the subregion
                subregion_mask = blur_mask[y:y+h, x:x+w]

                # Blur the subregion
                blurred_subregion = cv2.blur(subregion, (blur_ratio, blur_ratio))

                # Combine the blurred subregion with the original image using the mask
                subregion_with_blur = np.where(subregion_mask[..., None] == 255, blurred_subregion, subregion)

                # Replace the subregion in the original image with the blurred subregion
                im0[y:y+h, x:x+w] = subregion_with_blur

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
