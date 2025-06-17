# upload_video.py
import av
import os
import sys
import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose, draw_text, get_landmark_array
from process_frame import ProcessFrame as ProcessFrameSquat
from process_frame_pushup import ProcessFramePushup
from thresholds import get_thresholds_beginner, get_thresholds_pro, get_thresholds_pushup

st.title('AI Fitness Trainer with Action Recognition')

# --- LSTM Configuration ---
LSTM_MODEL_PATH = "pushup_squat_lstm_model.h5"
SEQUENCE_LENGTH = 10
NUM_LANDMARKS_LSTM = 33
NUM_COORDINATES_LSTM = 4
LSTM_EXPECTED_NUM_FEATURES = NUM_LANDMARKS_LSTM * NUM_COORDINATES_LSTM

ACTION_LABELS = {
    1: "PUSHUP",
    0: "SQUAT"
}
PREDICTION_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_ACTION = "DETECTING..."
UNCERTAIN_ACTION = "UNCERTAIN" 

# Load LSTM model
@st.cache_resource
def load_lstm_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"LSTM Model loaded successfully from {model_path}")
        try:
            model_input_shape = model.input_shape
            st.info(f"Model expected input shape: {model_input_shape}")
        except Exception as e:
            st.warning(f"Could not determine model input shape automatically: {e}")
        return model
    except Exception as e:
        st.error(f"Error loading LSTM model from {model_path}: {e}")
        st.warning("Action recognition will be disabled.")
        return None

lstm_model = load_lstm_model(LSTM_MODEL_PATH)
if lstm_model:
    try:
        LSTM_EXPECTED_NUM_FEATURES_FROM_MODEL = lstm_model.input_shape[-1]
        if LSTM_EXPECTED_NUM_FEATURES_FROM_MODEL != LSTM_EXPECTED_NUM_FEATURES:
            st.warning(f"LSTM_EXPECTED_NUM_FEATURES mismatch! From model: {LSTM_EXPECTED_NUM_FEATURES_FROM_MODEL}, Configured: {LSTM_EXPECTED_NUM_FEATURES}. Using value from model.")
            LSTM_EXPECTED_NUM_FEATURES = LSTM_EXPECTED_NUM_FEATURES_FROM_MODEL
    except:
        st.warning("Could not verify LSTM_EXPECTED_NUM_FEATURES from loaded model. Using configured value.")

# --- UI Elements ---
st.sidebar.header("Input Source")
input_source = st.sidebar.radio("Select input source:", ("Upload Video", "Real-time Camera"))

squat_mode_radio = st.sidebar.radio('Select Squat Mode (if squats are detected)',
                            ['Beginner', 'Pro'], horizontal=True, key='squat_mode_selection')

# MediaPipe Pose Detector
pose_detector = get_mediapipe_pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_pose_solution = mp.solutions.pose

# --- Session State Initialization ---
if 'download' not in st.session_state:
    st.session_state['download'] = False
if 'current_action' not in st.session_state:
    st.session_state['current_action'] = DEFAULT_ACTION
if 'action_processor' not in st.session_state:
    st.session_state['action_processor'] = None
if 'sequence_data' not in st.session_state:
    st.session_state['sequence_data'] = deque(maxlen=SEQUENCE_LENGTH)
if 'run_camera' not in st.session_state:
    st.session_state['run_camera'] = False
if 'camera_started_once' not in st.session_state: 
    st.session_state['camera_started_once'] = False


# --- Helper Functions ---
def prepare_lstm_input_from_landmarks(pose_landmarks_list):
    features = []
    if pose_landmarks_list:
        for lm in pose_landmarks_list.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])

    if len(features) < LSTM_EXPECTED_NUM_FEATURES:
        padding = [0.0] * (LSTM_EXPECTED_NUM_FEATURES - len(features))
        features.extend(padding)
    elif len(features) > LSTM_EXPECTED_NUM_FEATURES:
        features = features[:LSTM_EXPECTED_NUM_FEATURES]

    return np.array(features, dtype=np.float32)

def reset_app_state():
    """Resets states related to processing and LSTM."""
    st.session_state['current_action'] = DEFAULT_ACTION
    st.session_state['action_processor'] = None
    st.session_state['sequence_data'].clear()

def process_and_display_frame(frame_rgb, frame_width, frame_height):
    """Processes a single frame for pose, LSTM, and exercise analysis."""
    display_frame = frame_rgb.copy()
    keypoints_results = pose_detector.process(frame_rgb)

    current_lstm_features = prepare_lstm_input_from_landmarks(keypoints_results.pose_landmarks)
    st.session_state['sequence_data'].append(current_lstm_features)

    new_recognized_action_this_frame = st.session_state['current_action']
    current_confidence_for_display = 0.0

    if len(st.session_state['sequence_data']) == SEQUENCE_LENGTH and lstm_model:
        lstm_input_array = np.expand_dims(np.array(list(st.session_state['sequence_data']), dtype=np.float32), axis=0)

        if lstm_input_array.shape == (1, SEQUENCE_LENGTH, LSTM_EXPECTED_NUM_FEATURES):
            prediction_value = lstm_model.predict(lstm_input_array, verbose=0)[0][0]

            predicted_label_id = -1
            confidence_val = 0.0

            if prediction_value > PREDICTION_CONFIDENCE_THRESHOLD:
                predicted_label_id = 1 
                confidence_val = prediction_value
            elif (1 - prediction_value) > PREDICTION_CONFIDENCE_THRESHOLD:
                predicted_label_id = 0 
                confidence_val = 1 - prediction_value
            else:
                confidence_val = max(prediction_value, 1 - prediction_value)

            current_confidence_for_display = confidence_val 

            if predicted_label_id in ACTION_LABELS:
                new_recognized_action_this_frame = ACTION_LABELS[predicted_label_id]
            else:
                if st.session_state['current_action'] != DEFAULT_ACTION and st.session_state['current_action'] != UNCERTAIN_ACTION :
                    new_recognized_action_this_frame = st.session_state['current_action']
                else:
                    new_recognized_action_this_frame = UNCERTAIN_ACTION

            if new_recognized_action_this_frame != st.session_state['current_action']:
                st.session_state['current_action'] = new_recognized_action_this_frame
                if st.session_state['current_action'] == "SQUAT":
                    squat_mode = squat_mode_radio
                    thresholds = get_thresholds_beginner() if squat_mode == 'Beginner' else get_thresholds_pro()
                    st.session_state['action_processor'] = ProcessFrameSquat(thresholds=thresholds)
                    # st.toast(f"Action: SQUAT ({squat_mode}) - Conf: {confidence_val:.2f}")
                elif st.session_state['current_action'] == "PUSHUP":
                    thresholds = get_thresholds_pushup()
                    st.session_state['action_processor'] = ProcessFramePushup(thresholds=thresholds)
                    # st.toast(f"Action: PUSHUP - Conf: {confidence_val:.2f}")
                else:
                    st.session_state['action_processor'] = None
                    # st.toast(f"Action: {st.session_state['current_action']}")
        else:
            st.sidebar.error(f"LSTM input shape error! Expected (1, {SEQUENCE_LENGTH}, {LSTM_EXPECTED_NUM_FEATURES}), got {lstm_input_array.shape}")

    sound_to_play = None
    if st.session_state['action_processor'] and keypoints_results.pose_landmarks:
        display_frame, sound_to_play = st.session_state['action_processor'].process(display_frame, keypoints_results)
    elif keypoints_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display_frame, keypoints_results.pose_landmarks,
            mp_pose_solution.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    action_text_pos = (30, 30)
    action_text_color = (255, 255, 0)
    action_text_bg_color = (0, 0, 128)

    action_display_text = f"Action: {st.session_state['current_action'].upper()}"
    if st.session_state['current_action'] not in [DEFAULT_ACTION, UNCERTAIN_ACTION]:
         action_display_text += f" ({current_confidence_for_display:.2f})"

    draw_text(display_frame, action_display_text,
              pos=action_text_pos, font_scale=0.7,
              text_color=action_text_color, text_color_bg=action_text_bg_color,
              font_thickness=2, box_offset=(15,8))

    return display_frame


# --- Main Logic ---
stframe = st.empty() 
download_button_placeholder = st.empty() 

if input_source == "Upload Video":
    st.session_state['run_camera'] = False

    output_video_file = f'output_fitness_analysis.mp4'
    if os.path.exists(output_video_file):
        try:
            os.remove(output_video_file)
        except PermissionError:
            st.warning(f"Could not remove existing output file {output_video_file}. It might be in use.")


    with st.form('Upload_Video_Form', clear_on_submit=True):
        up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'], key='video_uploader')
        uploaded = st.form_submit_button("Process Video")

    if up_file and uploaded:
        reset_app_state() 
        download_button_placeholder.empty()
        st.session_state['download'] = False

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            tfile.write(up_file.read())
            tfile.flush()

            vf = cv2.VideoCapture(tfile.name)
            if not vf.isOpened():
                st.error(f"Error: Could not open video file: {tfile.name}")
            else:
                fps = int(vf.get(cv2.CAP_PROP_FPS)) if vf.get(cv2.CAP_PROP_FPS) > 0 else 30
                width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

                ip_vid_str = "### Input Video Preview"
                st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)
                ip_video = st.sidebar.video(tfile.name)

                progress_bar = st.progress(0)
                total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0

                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break

                    processed_frames += 1
                    if total_frames > 0 :
                        progress_bar.progress(min(1.0, processed_frames / total_frames))

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_display_frame = process_and_display_frame(frame_rgb, width, height)

                    stframe.image(processed_display_frame, channels="RGB")
                    video_output.write(cv2.cvtColor(processed_display_frame, cv2.COLOR_RGB2BGR))

                vf.release()
                video_output.release()
                stframe.empty() 
                if 'ip_video' in locals(): ip_video.empty()
                progress_bar.empty()
                st.success("Video processing complete!")
                st.session_state['download'] = True 

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
            if 'vf' in locals() and vf and vf.isOpened(): vf.release()
            if 'video_output' in locals() and video_output and video_output.isOpened(): video_output.release()
        finally:
            if 'tfile' in locals() and tfile:
                tfile.close()
                if os.path.exists(tfile.name):
                    try:
                        os.unlink(tfile.name)
                    except Exception as e_unlink:
                        st.warning(f"Could not delete temp file {tfile.name}: {e_unlink}")
    
    if st.session_state['download'] and os.path.exists(output_video_file):
        with open(output_video_file, 'rb') as op_vid:
            download_button_placeholder.download_button('Download Processed Video', data = op_vid, file_name=output_video_file)

elif input_source == "Real-time Camera":
    st.session_state['download'] = False 
    download_button_placeholder.empty() 

    if not st.session_state['camera_started_once']:
        reset_app_state() 
        st.session_state['camera_started_once'] = True


    if st.sidebar.button("Start Camera", key="start_cam_btn"):
        st.session_state['run_camera'] = True
        reset_app_state() 
        st.sidebar.info("Camera is ON. Click 'Stop Camera' to turn off.")

    if st.sidebar.button("Stop Camera", key="stop_cam_btn"):
        st.session_state['run_camera'] = False
        st.sidebar.info("Camera is OFF.")
        stframe.empty()

    if st.session_state['run_camera']:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera. Please check if it's connected and not in use by another application.")
            st.session_state['run_camera'] = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while st.session_state['run_camera']:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from camera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb_mirrored = cv2.flip(frame_rgb, 1)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                processed_display_frame = process_and_display_frame(frame_rgb_mirrored, frame_width, frame_height)

                stframe.image(processed_display_frame, channels="RGB")

            cap.release()
            if not st.session_state['run_camera']:
                 stframe.empty()
    else:
        if st.session_state['camera_started_once']:
            st.info("Camera is off. Click 'Start Camera' in the sidebar.")
        else:
            st.info("Select 'Real-time Camera' and click 'Start Camera' in the sidebar to begin.")