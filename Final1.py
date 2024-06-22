import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load YOLO model
model = YOLO("best.pt")  # Use the segmentation model

# Get the class index for 'female'
person_class_index = None
for idx, name in model.names.items():
    if name == 'female':
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

                # Adjust bounding box to ensure it doesn't go out of frame boundaries
                x_end = min(x + w, im0.shape[1])
                y_end = min(y + h, im0.shape[0])

                # Create a subregion for blurring
                subregion = im0[y:y_end, x:x_end]

                # Create a mask for the subregion
                subregion_mask = blur_mask[y:y_end, x:x_end]

                # Blur the subregion
                blurred_subregion = cv2.blur(subregion, (blur_ratio, blur_ratio))

                # Ensure the shapes match
                if subregion.shape[:2] == subregion_mask.shape:
                    # Combine the blurred subregion with the original image using the mask
                    subregion_with_blur = np.where(subregion_mask[..., None] == 255, blurred_subregion, subregion)

                    # Replace the subregion in the original image with the blurred subregion
                    im0[y:y_end, x:x_end] = subregion_with_blur

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
