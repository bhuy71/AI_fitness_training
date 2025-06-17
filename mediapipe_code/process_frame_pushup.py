# process_frame_pushup.py
import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_array, draw_text
import mediapipe as mp 

class ProcessFramePushup:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.radius = 20
        self.COLORS = {
            'blue': (0, 127, 255), 'red': (255, 50, 50), 'green': (0, 255, 127),
            'light_green': (100, 233, 127), 'yellow': (255, 255, 0), 'magenta': (255, 0, 255),
            'white': (255, 255, 255), 'cyan': (0, 255, 255), 'light_blue': (102, 204, 255)
        }
        self.feature_indices = {
            'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24, 'left_ear': 7, 'right_ear': 8,
            'left_knee': 25, 'right_knee': 26, 
            'left_ankle': 27, 'right_ankle': 28, 
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        self.FEEDBACK_ID_MAP = {
            0: ('YOUR BACK IS NOT STRAIGHT', 220, (200, 100, 0)),
            1: ('STRAIGHTEN ARMS AT TOP', 180, (0, 153, 255)),
            2: ('ARMS NOT PERPENDICULAR AT BOTTOM', 140, (255, 80, 80)), 
        }
        self.reset_state()

    def reset_state(self):
        self.state_tracker = {
            'state_seq': [], 'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(), 'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'DISPLAY_TEXT': np.full((3,), False), 
            'COUNT_FRAMES': np.zeros((3,), dtype=np.int64),
            'INCORRECT_POSTURE': False,
            'prev_state': None, 'curr_state': None, 
            'PUSHUP_COUNT': 0, 'IMPROPER_PUSHUP': 0,
            'went_low_enough_correctly': False 
        }

    def _get_pushup_state(self, elbow_angle): 
        if self.thresholds['ELBOW_ANGLE']['STRAIGHT'][0] <= elbow_angle <= self.thresholds['ELBOW_ANGLE']['STRAIGHT'][1]:
            return 's1' 
        elif self.thresholds['ELBOW_ANGLE']['BENT_LOW'][0] <= elbow_angle <= self.thresholds['ELBOW_ANGLE']['BENT_LOW'][1]:
            return 's3' 
        elif self.thresholds['ELBOW_ANGLE']['BENDING'][0] <= elbow_angle <= self.thresholds['ELBOW_ANGLE']['BENDING'][1]:
            return 's2' 
        return None
    
    def _show_feedback(self, frame, c_frame_flags, dict_maps):
        for idx in np.where(c_frame_flags)[0]:
            if idx in dict_maps:
                draw_text(frame, dict_maps[idx][0], pos=(30, dict_maps[idx][1]),
                          text_color=(255, 255, 230), font_scale=0.6, text_color_bg=dict_maps[idx][2])
        return frame

    def process(self, frame: np.array, keypoints_results):
        play_sound = None
        frame_height, frame_width, _ = frame.shape

        if keypoints_results.pose_landmarks:
            ps_lm_list = keypoints_results.pose_landmarks.landmark 

            nose_coord_offset = get_landmark_array(ps_lm_list, self.feature_indices['nose'], frame_width, frame_height)
            
            left_shldr_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_shoulder'], frame_width, frame_height)
            right_shldr_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_shoulder'], frame_width, frame_height)
            left_elbow_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_elbow'], frame_width, frame_height)
            right_elbow_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_elbow'], frame_width, frame_height)
            left_wrist_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_wrist'], frame_width, frame_height)
            right_wrist_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_wrist'], frame_width, frame_height)
            left_hip_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_hip'], frame_width, frame_height)
            right_hip_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_hip'], frame_width, frame_height)
            left_ear_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_ear'], frame_width, frame_height)
            right_ear_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_ear'], frame_width, frame_height)
            left_knee_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_knee'], frame_width, frame_height)
            right_knee_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_knee'], frame_width, frame_height)
            left_ankle_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_ankle'], frame_width, frame_height)
            right_ankle_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_ankle'], frame_width, frame_height)
            left_foot_coord = get_landmark_array(ps_lm_list, self.feature_indices['left_foot_index'], frame_width, frame_height)
            right_foot_coord = get_landmark_array(ps_lm_list, self.feature_indices['right_foot_index'], frame_width, frame_height)

            shldr_coord_main = left_shldr_coord
            elbow_coord_main = left_elbow_coord
            wrist_coord_main = left_wrist_coord
            hip_coord_main = left_hip_coord 
            ear_coord_main = left_ear_coord   

            if ps_lm_list[self.feature_indices['left_shoulder']].visibility < 0.5 and \
               ps_lm_list[self.feature_indices['right_shoulder']].visibility > 0.5:
                shldr_coord_main = right_shldr_coord
                elbow_coord_main = right_elbow_coord
                wrist_coord_main = right_wrist_coord
                hip_coord_main = right_hip_coord
                ear_coord_main = right_ear_coord

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord_offset)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                display_inactivity = False; end_time = time.perf_counter(); self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']; self.state_tracker['start_inactive_time_front'] = end_time
                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']: self.state_tracker['PUSHUP_COUNT'] = 0; self.state_tracker['IMPROPER_PUSHUP'] = 0; display_inactivity = True
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1); cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)
                if self.flip_frame: frame = cv2.flip(frame, 1)
                if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0; self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                draw_text(frame, "CORRECT: " + str(self.state_tracker['PUSHUP_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_PUSHUP']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                draw_text(frame, 'CAMERA NOT ALIGNED PROPERLY!!!', pos=(30, frame_height-60), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                draw_text(frame, 'OFFSET ANGLE: '+str(offset_angle), pos=(30, frame_height-30), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0; self.state_tracker['prev_state'] =  None; self.state_tracker['curr_state'] = None; self.state_tracker['went_low_enough_correctly'] = False
            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                angle_elbow_main = find_angle(shldr_coord_main, wrist_coord_main, elbow_coord_main)
                angle_ear_shoulder_hip_main = find_angle(ear_coord_main, hip_coord_main, shldr_coord_main)

                cv2.line(frame, left_shldr_coord, left_elbow_coord, self.COLORS['light_blue'], 4)
                cv2.line(frame, left_elbow_coord, left_wrist_coord, self.COLORS['light_blue'], 4)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, left_elbow_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, left_wrist_coord, 7, self.COLORS['yellow'], -1)

                cv2.line(frame, right_shldr_coord, right_elbow_coord, self.COLORS['light_blue'], 4)
                cv2.line(frame, right_elbow_coord, right_wrist_coord, self.COLORS['light_blue'], 4)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_elbow_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_wrist_coord, 7, self.COLORS['yellow'], -1)

                cv2.line(frame, shldr_coord_main, hip_coord_main, self.COLORS['light_blue'], 4) 
                cv2.line(frame, ear_coord_main, shldr_coord_main, self.COLORS['light_blue'], 4)  
                cv2.circle(frame, hip_coord_main, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, ear_coord_main, 7, self.COLORS['yellow'], -1)
                
                if ps_lm_list[self.feature_indices['left_knee']].visibility > 0.1: 
                    cv2.line(frame, left_hip_coord, left_knee_coord, self.COLORS['light_blue'], 4)
                    cv2.line(frame, left_knee_coord, left_ankle_coord, self.COLORS['light_blue'], 4)
                    cv2.line(frame, left_ankle_coord, left_foot_coord, self.COLORS['light_blue'], 4)
                    cv2.circle(frame, left_knee_coord, 7, self.COLORS['yellow'], -1)
                    cv2.circle(frame, left_ankle_coord, 7, self.COLORS['yellow'], -1)
                    cv2.circle(frame, left_foot_coord, 7, self.COLORS['yellow'], -1)

                if ps_lm_list[self.feature_indices['right_knee']].visibility > 0.1:
                    cv2.line(frame, right_hip_coord, right_knee_coord, self.COLORS['light_blue'], 4)
                    cv2.line(frame, right_knee_coord, right_ankle_coord, self.COLORS['light_blue'], 4)
                    cv2.line(frame, right_ankle_coord, right_foot_coord, self.COLORS['light_blue'], 4)
                    cv2.circle(frame, right_knee_coord, 7, self.COLORS['yellow'], -1)
                    cv2.circle(frame, right_ankle_coord, 7, self.COLORS['yellow'], -1)
                    cv2.circle(frame, right_foot_coord, 7, self.COLORS['yellow'], -1)

                current_state = self._get_pushup_state(int(angle_elbow_main)) 
                self.state_tracker['curr_state'] = current_state
                
                if current_state: 
                    if self.state_tracker['prev_state'] == 's1' and (current_state == 's2' or current_state == 's3'):
                        self.state_tracker['INCORRECT_POSTURE'] = False 
                        self.state_tracker['went_low_enough_correctly'] = False 
                    
                    if current_state == 's2' or current_state == 's3':
                        if not (self.thresholds.get('BACK_EAR_SHOULDER_HIP_MIN', 160) <= angle_ear_shoulder_hip_main <= self.thresholds.get('BACK_EAR_SHOULDER_HIP_MAX', 180)):
                            self.state_tracker['DISPLAY_TEXT'][0] = True 
                            self.state_tracker['INCORRECT_POSTURE'] = True
                    
                    if current_state == 's3':
                        if not (self.thresholds['ELBOW_ANGLE']['BENT_LOW'][0] <= angle_elbow_main <= self.thresholds['ELBOW_ANGLE']['BENT_LOW'][1]):
                            self.state_tracker['DISPLAY_TEXT'][2] = True 
                            self.state_tracker['INCORRECT_POSTURE'] = True
                            self.state_tracker['went_low_enough_correctly'] = False 
                        else: 
                            self.state_tracker['went_low_enough_correctly'] = True 
                            self.state_tracker['DISPLAY_TEXT'][2] = False 
                    
                    if (self.state_tracker['prev_state'] == 's3' or \
                        (self.state_tracker['prev_state'] == 's2' and self.state_tracker['went_low_enough_correctly'])) and \
                       current_state == 's1':
                        if not (self.thresholds['ELBOW_ANGLE']['STRAIGHT'][0] <= angle_elbow_main <= self.thresholds['ELBOW_ANGLE']['STRAIGHT'][1]):
                            self.state_tracker['DISPLAY_TEXT'][1] = True 
                            self.state_tracker['INCORRECT_POSTURE'] = True
                        
                        if self.state_tracker['went_low_enough_correctly'] and not self.state_tracker['INCORRECT_POSTURE']:
                            self.state_tracker['PUSHUP_COUNT'] += 1
                        else: 
                            self.state_tracker['IMPROPER_PUSHUP'] += 1
                        self.state_tracker['went_low_enough_correctly'] = False 
                        self.state_tracker['INCORRECT_POSTURE'] = False 
                    
                    elif current_state == 's1' and self.state_tracker['prev_state'] == 's1':
                         if not (self.thresholds['ELBOW_ANGLE']['STRAIGHT'][0] <= angle_elbow_main <= self.thresholds['ELBOW_ANGLE']['STRAIGHT'][1]):
                            self.state_tracker['DISPLAY_TEXT'][1] = True 

                display_inactivity = False
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state'] and current_state is not None : 
                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time
                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['PUSHUP_COUNT'] = 0; self.state_tracker['IMPROPER_PUSHUP'] = 0; display_inactivity = True
                else:
                    self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0

                shldr_text_coord_x = shldr_coord_main[0] + 10
                shldr_text_coord_y = shldr_coord_main[1] - 10
                elbow_text_coord_x = elbow_coord_main[0] + 10
                elbow_text_coord_y = elbow_coord_main[1]

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    shldr_text_coord_x = frame_width - shldr_coord_main[0] + 10
                    elbow_text_coord_x = frame_width - elbow_coord_main[0] + 10
                
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(frame, self.state_tracker['DISPLAY_TEXT'], self.FEEDBACK_ID_MAP)

                if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0
                
                cv2.putText(frame, f"Elbow: {int(angle_elbow_main)}", (elbow_text_coord_x, elbow_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2)
                cv2.putText(frame, f"Back: {int(angle_ear_shoulder_hip_main)}", (shldr_text_coord_x, shldr_text_coord_y), self.font, 0.6, self.COLORS['cyan'], 2)
                
                draw_text(frame, "CORRECT: " + str(self.state_tracker['PUSHUP_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_PUSHUP']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
                if current_state: 
                    self.state_tracker['prev_state'] = current_state
        else:
            if self.flip_frame: frame = cv2.flip(frame, 1)
            end_time = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']; display_inactivity = False
            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']: self.state_tracker['PUSHUP_COUNT'] = 0; self.state_tracker['IMPROPER_PUSHUP'] = 0; display_inactivity = True
            self.state_tracker['start_inactive_time'] = end_time
            draw_text(frame, "CORRECT: " + str(self.state_tracker['PUSHUP_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
            draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_PUSHUP']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
            if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0
            self.state_tracker['prev_state'] =  None; self.state_tracker['curr_state'] = None; self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0; self.state_tracker['INCORRECT_POSTURE'] = False; self.state_tracker['went_low_enough_correctly'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((3,), False); self.state_tracker['COUNT_FRAMES'] = np.zeros((3,), dtype=np.int64); self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound