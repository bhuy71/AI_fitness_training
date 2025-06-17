import cv2
import mediapipe as mp
import pandas as pd
import os

# --- Configuration ---
DATA_DIR = "data_lstm"
ACTIONS = ["pushup", "squat"] 
OUTPUT_DIR = "."

SHOW_VIDEO_PROCESSING = True


mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    """
    Trích xuất tọa độ (x, y, z, visibility) từ kết quả nhận diện pose.
    Trả về một list các giá trị landmark cho một frame.
    """
    c_lm = []
    if results.pose_landmarks: 
        for lm in results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(image, results):

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

# --- Xử lý chính ---
for action in ACTIONS:
    print(f"Processing action: {action}")
    action_landmarks_list = [] 
    action_folder_path = os.path.join(DATA_DIR, action)

    if not os.path.isdir(action_folder_path):
        print(f"  Warning: Folder for action '{action}' not found at '{action_folder_path}'. Skipping.")
        continue

 
    video_files = [f for f in os.listdir(action_folder_path) if f.lower().endswith(".mp4")]

    if not video_files:
        print(f"  Warning: No .mp4 files found in '{action_folder_path}' for action '{action}'. Skipping.")
        continue

    for video_file in video_files:
        video_path = os.path.join(action_folder_path, video_file)
        print(f"  Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Could not open video {video_path}")
            continue

        frame_count_this_video = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False 


            results = pose.process(image_rgb)

            image_rgb.flags.writeable = True 
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 

            if results.pose_landmarks:
                landmarks = make_landmark_timestep(results)
                if landmarks: 
                    action_landmarks_list.append(landmarks)
                    frame_count_this_video += 1

                if SHOW_VIDEO_PROCESSING:
                    annotated_image = draw_landmark_on_image(image_bgr.copy(), results)
                    cv2.imshow(f'MediaPipe Pose - {action} - {video_file}', annotated_image)
            else:
                if SHOW_VIDEO_PROCESSING:
                    cv2.imshow(f'MediaPipe Pose - {action} - {video_file}', image_bgr)


            if SHOW_VIDEO_PROCESSING and cv2.waitKey(5) & 0xFF == ord('q'):
                print("    User interrupted video processing.")
                cap.release()
                break
        
        print(f"    Extracted landmarks from {frame_count_this_video} frames in '{video_file}'.")
        cap.release()
        if SHOW_VIDEO_PROCESSING and (cv2.waitKey(1) & 0xFF == ord('q')): 
             break 
    
    if SHOW_VIDEO_PROCESSING and (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    if action_landmarks_list:
        df = pd.DataFrame(action_landmarks_list)
        output_file_path = os.path.join(OUTPUT_DIR, f"{action}.txt")
        df.to_csv(output_file_path, header=False, index=False, sep=',')
        print(f"Saved data for {action} to {output_file_path} ({len(action_landmarks_list)} frames total)")
    else:
        print(f"No landmarks collected for action {action}. No file saved.")

# Dọn dẹp
if SHOW_VIDEO_PROCESSING:
    cv2.destroyAllWindows()
pose.close() 
print("Processing complete.")