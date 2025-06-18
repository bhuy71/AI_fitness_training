from tensorflow.keras.models import load_model
import numpy as np

import cv2
import numpy as np
import sys

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

# ====================== Utilities ======================
def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
            return np.nan
        ab = a - b
        cb = c - b
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print("Angle calculation error:", e)
        return np.nan

def calculate_vector_angle(v1, v2):
    try:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return np.nan
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print("Vector angle calc error:", e)
        return np.nan

def draw_vertical_dashed_line(frame, x, y_center, length=60, color=(255, 255, 255), thickness=2, gap=8):
    y_start = int(y_center - length / 2)
    y_end = int(y_center + length / 2)
    for y in range(y_start, y_end, gap * 2):
        cv2.line(frame, (x, y), (x, min(y + gap, y_end)), color, thickness)

def safe_int_point(pt):
    if pt is None or np.isnan(pt).any():
        return None
    return tuple(map(int, pt))


def is_valid_point(pt):
    return pt is not None and not np.isnan(pt[0]) and not np.isnan(pt[1])

# ====================== Exercise Counter ======================
class ExerciseCounter:
    def __init__(self, exercise="squat", debounce_threshold=2):
        self.exercise = exercise
        self.state = "up"
        self.last_state = "up"
        self.frame_counter = 0
        self.min_hold_frames = 8
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.ready_to_count = False
        self.was_form_good_in_down = False  # ✅ track form quality in down position
        if exercise == "pushup":
            self.debounce = 0
            self.down_hold_frames = 0
            self.debounce_threshold = debounce_threshold

    def update(self, keypoints, frame=None):
        if self.exercise == "squat":
            try:
                hip = np.array(keypoints[11], dtype=np.float32)
                knee = np.array(keypoints[13], dtype=np.float32)
                ankle = np.array(keypoints[15], dtype=np.float32)
                shoulder = np.array(keypoints[5], dtype=np.float32)
                ear = np.array(keypoints[3], dtype=np.float32)

                vertical = np.array([0, -1], dtype=np.float32)
                thigh_vec = hip - knee
                back_vec = shoulder - hip
                torso_vec = hip - shoulder
                neck_vec = ear - shoulder

                knee_angle = calculate_vector_angle(thigh_vec, vertical)
                hip_angle = calculate_vector_angle(back_vec, vertical)
                shoulder_angle = calculate_vector_angle(torso_vec, neck_vec)

                good_knee = not np.isnan(knee_angle) and knee_angle < 110
                good_hip = not np.isnan(hip_angle) and hip_angle < 110
                good_back = good_hip
                good_shoulder = not np.isnan(shoulder_angle) and shoulder_angle > 140

                all_good = good_knee and good_hip and good_back and good_shoulder

                # Transition from up to down (form must be good)
                if all_good:
                    if self.state == "up":
                        self.frame_counter += 1
                        if self.frame_counter >= self.min_hold_frames:
                            self.state = "down"
                            self.ready_to_count = True
                            self.was_form_good_in_down = True  # ✅ save form state
                            self.frame_counter = 0
                    else:
                        self.frame_counter = 0
                else:
                    if self.state == "down":
                        self.frame_counter += 1
                        if self.frame_counter >= self.min_hold_frames:
                            self.state = "up"
                            if self.ready_to_count:
                                if self.was_form_good_in_down:  # ✅ use cached form status
                                    self.correct_reps += 1
                                    print(">>> Correct rep counted (+1)")
                                else:
                                    self.incorrect_reps += 1
                                    print(">>> Incorrect rep counted (+1)")
                                self.ready_to_count = False
                            self.frame_counter = 0
                            self.was_form_good_in_down = False  # reset
                    else:
                        self.frame_counter = 0

                if frame is not None:
                    try:
                        knee_int = safe_int_point(knee)
                        hip_int = safe_int_point(hip)
                        shoulder_int = safe_int_point(shoulder)

                        if hip_int:
                            draw_vertical_dashed_line(frame, hip_int[0], hip_int[1], length=80)
                        if knee_int:
                            draw_vertical_dashed_line(frame, knee_int[0], knee_int[1], length=80)

                        if knee_int is not None and not np.isnan(knee_angle):
                            cv2.putText(frame, f"{int(knee_angle)}°", (knee_int[0] + 30, knee_int[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 50), 2)

                        if hip_int is not None and not np.isnan(hip_angle):
                            cv2.putText(frame, f"{int(hip_angle)}°", (hip_int[0] - 40, hip_int[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if shoulder_int is not None and not np.isnan(shoulder_angle):
                            cv2.putText(frame, f"{int(shoulder_angle)}°", (shoulder_int[0], shoulder_int[1] - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

                        warning_y = 210
                        if not good_knee:
                            cv2.putText(frame, "Lower your hips", (50, warning_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
                            warning_y += 40
                        if not good_hip:
                            cv2.putText(frame, "Bend more at your hips", (50, warning_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            warning_y += 40
                        if not good_back:
                            cv2.putText(frame, "Keep your back straight", (50, warning_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            warning_y += 40
                        if not good_shoulder:
                            cv2.putText(frame, "Straighten your upper body", (50, warning_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    except Exception as e:
                        print("Draw error:", e)
            except Exception as e:
                print("Squat detection error:", e)

        elif self.exercise == "pushup":
            try:
                angle_l = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
                angle_r = calculate_angle(keypoints[6], keypoints[8], keypoints[10])

                if np.isnan(angle_l) and np.isnan(angle_r):
                    print("[PUSHUP] Elbows not visible")
                    return

                avg_elbow_angle = np.nanmean([angle for angle in [angle_l, angle_r] if not np.isnan(angle)])

                hip = keypoints[11]
                ankle = keypoints[15]
                ear = keypoints[3]

                body_alignment_angle = calculate_angle(ankle, hip, ear)
                is_body_straight = body_alignment_angle > 155
                is_arm_straight = avg_elbow_angle > 155
                is_arm_bent = avg_elbow_angle < 90

                print(f"[PUSHUP] L: {angle_l:.2f}, R: {angle_r:.2f}, Avg: {avg_elbow_angle:.2f}, Debounce: {self.debounce}, State: {self.state}")

                if self.state == "up":
                    if is_arm_bent:
                        self.debounce += 1
                        if self.debounce >= self.debounce_threshold:
                            self.state = "down"
                            self.debounce = 0
                            self.down_hold_frames = 0
                            print(">>> Transition to DOWN")
                    else:
                        self.debounce = 0
                        self.down_hold_frames = 0

                elif self.state == "down":
                    if is_arm_straight:
                        self.debounce += 1
                        if self.debounce >= self.debounce_threshold:
                            self.state = "up"
                            self.debounce = 0
                            self.down_hold_frames = 0

                            if is_body_straight and is_arm_straight:
                                self.correct_reps += 1
                                print(">>> Correct rep counted (+1)")
                            else:
                                self.incorrect_reps += 1
                                print(">>> Incorrect rep counted (+1)")
                    else:
                        self.debounce = 0
                        self.down_hold_frames += 1
                        if self.down_hold_frames > 90:
                            self.incorrect_reps += 1
                            self.state = "up"
                            self.down_hold_frames = 0
                            print(">>> Forced transition to UP (Incorrect +1)")

                if frame is not None:
                    elbow_l_pos = tuple(map(int, keypoints[7])) if not np.isnan(angle_l) else None
                    elbow_r_pos = tuple(map(int, keypoints[8])) if not np.isnan(angle_r) else None

                    label = ""
                    color = (0, 255, 0)

                    if self.state == "up":
                        if is_arm_straight:
                            label = "Arm Straight"
                            color = (0, 255, 0)
                        else:
                            label = "Not Straight"
                            color = (0, 0, 255)
                    elif self.state == "down":
                        if is_arm_bent:
                            label = "Bent < 90°"
                            color = (0, 255, 0)
                        else:
                            label = "Not Bent"
                            color = (0, 0, 255)

                    if elbow_l_pos and is_valid_point(elbow_l_pos):
                        cv2.putText(frame, f"{int(angle_l)}°", elbow_l_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if elbow_r_pos and is_valid_point(elbow_r_pos):
                        cv2.putText(frame, f"{int(angle_r)}°", elbow_r_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                    if is_valid_point(hip) and not np.isnan(body_alignment_angle):
                        hip_int = tuple(map(int, hip))
                        cv2.putText(frame, f"{int(body_alignment_angle)}°", hip_int,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    feedback_y = 150
                    if not is_arm_straight and self.state == "up":
                        cv2.putText(frame, "Bend your elbows more", (50, feedback_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        feedback_y += 40
                    if not is_arm_bent and self.state == "down":
                        cv2.putText(frame, "Straighten your arms", (50, feedback_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        feedback_y += 40
                    if not is_body_straight and self.state == "up":
                        cv2.putText(frame, "Keep your back straight", (50, feedback_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)


            except Exception as e:
                print("Push-up detection error:", e)

# ====================== Setup ======================
exercise_mode = sys.argv[1] if len(sys.argv) > 1 else "squat"
video_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\vuvie\Downloads\default_video.mp4"

cap = cv2.VideoCapture(video_path)
start_time = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

model_path = "models/hrnet_pose.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.3)

person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path, conf_thres=0.3)

lstm_model = load_model("models/pushup_squat_lstm_model.h5")  # Change path if needed
SEQUENCE_LENGTH = 30
sequence_buffer = []
prediction_label = None  # Final label from LSTM
label_displayed = False  # Flag to prevent re-predicting

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
frame_num = 0

debounce_val = 3 if exercise_mode == "pushup" else 5
exercise_counter = ExerciseCounter(exercise=exercise_mode, debounce_threshold=debounce_val)

# ====================== Main Loop ======================
while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        ret, frame = cap.read()
        if frame_num < start_time * 30:
            frame_num += 1
            continue
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    detections = person_detector(frame)
    ret, person_detections = filter_person_detections(detections)
    person_detector.boxes, person_detector.scores, person_detector.class_ids = person_detections

    if ret:
        total_heatmap, peaks = hrnet(frame, person_detections)
        frame = hrnet.draw_pose(frame)
        frame = person_detector.draw_detections(frame, mask_alpha=0.15)

        for person_keypoints in peaks:
            try:
                if len(person_keypoints[0]) == 3:
                    keypoints = [(int(x), int(y)) if not np.isnan(x) and not np.isnan(y) else (np.nan, np.nan)
                                 for x, y, _ in person_keypoints]
                else:
                    keypoints = [(int(x), int(y)) if not np.isnan(x) and not np.isnan(y) else (np.nan, np.nan)
                                 for x, y in person_keypoints]
            except Exception as e:
                print("Error processing keypoints:", e)
                continue

            if len(keypoints) >= 17:
                flattened = []
                for (x, y) in keypoints[:17]: 
                    try:
                        x_val = float(x)
                        x_val = x_val if not np.isnan(x_val) else 0
                    except:
                        x_val = 0

                    try:
                        y_val = float(y)
                        y_val = y_val if not np.isnan(y_val) else 0
                    except:
                        y_val = 0

                    flattened.extend([x_val, y_val])

                if not label_displayed:
                    flattened = []
                    for (x, y) in keypoints[:17]:
                        try:
                            x_val = float(x)
                            x_val = x_val if not np.isnan(x_val) else 0
                        except:
                            x_val = 0

                        try:
                            y_val = float(y)
                            y_val = y_val if not np.isnan(y_val) else 0
                        except:
                            y_val = 0

                        flattened.extend([x_val, y_val])
                    
                    sequence_buffer.append(flattened)

                    if len(sequence_buffer) == SEQUENCE_LENGTH:
                        input_seq = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, 34)
                        prediction = lstm_model.predict(input_seq, verbose=0)
                        prediction_label = "Pushup" if prediction[0][0] > 0.5 else "Squat"
                        print(f"[LSTM] Final Prediction: {prediction[0][0]:.2f} → {prediction_label}")
                        label_displayed = True  


                exercise_counter.update(keypoints, frame=frame)
                if exercise_mode == "pushup":
                    correct = getattr(exercise_counter, "correct_reps", 0)
                    incorrect = getattr(exercise_counter, "incorrect_reps", 0)
                    cv2.putText(frame, f"Correct Reps: {correct}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Incorrect Reps: {incorrect}", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, f"Correct Reps: {exercise_counter.correct_reps}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Incorrect Reps: {exercise_counter.incorrect_reps}", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    if prediction_label:
        cv2.putText(frame, f"Predicted: {prediction_label}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("Model Output", frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()