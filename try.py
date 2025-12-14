import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os
import sys

# --- FIX for old TensorFlow model formats ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# --- Configuration ---
MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"

# --- Load Model and Labels ---
model = load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]
print("Model and labels loaded.")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    sys.exit()

# <<< START OF RIGGING LOGIC >>>
override_label = None # This will store our forced prediction
# <<< END OF RIGGING LOGIC >>>

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Standard Image Processing ---
    img_resized = cv2.resize(frame, (224, 224))
    img_array = np.asarray(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    normalized_img_array = (img_array_expanded.astype(np.float32) / 127.5) - 1

    # --- Live Model Prediction ---
    preds = model.predict(normalized_img_array, verbose=0)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    try:
        class_name = labels[class_idx].split(' ', 1)[1]
    except IndexError:
        class_name = labels[class_idx]
    live_label_text = f"{class_name}: {confidence:.2%}"

    # --- KEYBOARD CONTROLS FOR OVERRIDE ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        override_label = " WET (100.0%)"
        print("Override set to WET")
    elif key == ord('d'):
        override_label = " DRY (100.0%)"
        print("Override set to DRY")
    elif key == ord('e'):
        override_label = " E-WASTE (100.0%)"
        print("Override set to E-WASTE")
    elif key == ord('c'):
        override_label = None # Clear the override
        #print("Override cleared. Using live model.")
    elif key == ord('q'):
        break # Quit the program

    # --- Display Logic ---
    # If an override is active, show it. Otherwise, show the live model's prediction.
    if override_label:
        final_label = override_label
        label_color = (0, 255, 255) # Yellow for overrides
    else:
        final_label = live_label_text
        label_color = (0, 255, 0) # Green for live predictions

    cv2.putText(frame, final_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
    cv2.putText(frame, "Keys: [w,d,e] to override, [c] to clear, [q] to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Waste Classifier", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()