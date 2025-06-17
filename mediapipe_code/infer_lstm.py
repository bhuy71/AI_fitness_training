import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import time

# --- Configuration ---
MODEL_PATH = "pushup_squat_lstm_model.h5"
VIDEO_SOURCE = "data_lstm\pushup\pushup_2.mp4"
NO_OF_TIMESTEPS = 10  

ACTION_LABELS = {
    1: "PUSHUP",
    0: "SQUAT"
}
CONFIDENCE_THRESHOLD = 0.7
DISPLAY_WIDTH = 800 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Load Model ---
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
    EXPECTED_NUM_FEATURES = model.input_shape[-1]
    print(f"Model expects {EXPECTED_NUM_FEATURES} features per timestep.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Hàm trợ giúp ---
def make_landmark_timestep(results):
    """Trích xuất landmarks (x, y, z, visibility) từ kết quả pose."""
    c_lm = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm

def draw_landmark_on_image(image, results):
    """Vẽ landmarks và connections lên ảnh."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

def draw_prediction_on_image(image, label, confidence):
    """Vẽ kết quả dự đoán lên ảnh."""
    cv2.putText(image, f"Action: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return image

# --- Xử lý Inference ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
    exit()

sequence_data = [] 
current_action = "Waiting..."
current_confidence = 0.0
frame_count = 0
fps = 0
start_time = time.time()

print("Starting inference... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = pose.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) 

  
    landmarks = make_landmark_timestep(results)


    if landmarks and len(landmarks) != EXPECTED_NUM_FEATURES:
        print(f"Warning: Number of extracted features ({len(landmarks)}) does not match model's expected features ({EXPECTED_NUM_FEATURES}). Skipping prediction for this frame.")
       
    elif landmarks: 
        sequence_data.append(landmarks)

    if len(sequence_data) > NO_OF_TIMESTEPS:
        sequence_data.pop(0) 

    if len(sequence_data) == NO_OF_TIMESTEPS:

        input_array = np.expand_dims(np.array(sequence_data, dtype=np.float32), axis=0)

        # Dự đoán
        prediction = model.predict(input_array, verbose=0)[0][0] 

        if prediction > CONFIDENCE_THRESHOLD: 
            predicted_label_id = 1
            current_confidence = prediction
        elif (1 - prediction) > CONFIDENCE_THRESHOLD: 
            predicted_label_id = 0
            current_confidence = 1 - prediction
        else:
            predicted_label_id = -1 
            current_confidence = max(prediction, 1-prediction)


        if predicted_label_id in ACTION_LABELS:
            current_action = ACTION_LABELS[predicted_label_id]
        else:
            current_action = "Uncertain"
    else:
        pass

    annotated_image = frame_bgr.copy()
    if results.pose_landmarks:
        annotated_image = draw_landmark_on_image(annotated_image, results)

    annotated_image = draw_prediction_on_image(annotated_image, current_action, current_confidence)

    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = time.time()
        frame_count = 0

    cv2.putText(annotated_image, f"FPS: {fps:.2f}", (annotated_image.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    h_orig, w_orig = annotated_image.shape[:2]
    if w_orig > DISPLAY_WIDTH:
        ratio = DISPLAY_WIDTH / float(w_orig)
        new_h = int(h_orig * ratio)
        resized_frame_to_display = cv2.resize(annotated_image, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_frame_to_display = annotated_image

    cv2.imshow('LSTM Action Recognition', resized_frame_to_display)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Inference finished.")