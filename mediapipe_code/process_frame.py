# process_frame.py
import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line,get_landmark_array

class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        self.flip_frame = flip_frame
        self.thresholds = thresholds 
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.radius = 20
        self.COLORS = {
                        'blue'       : (0, 127, 255), 'red': (255, 50, 50), 'green': (0, 255, 127),
                        'light_green': (100, 233, 127), 'yellow': (255, 255, 0), 'magenta': (255, 0, 255),
                        'white'      : (255,255,255), 'cyan': (0, 255, 255), 'light_blue' : (102, 204, 255)
                      }
        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 
                                'knee': 25, 'ankle': 27, 'foot': 31, 'ear': 7 
                             }
        self.right_features = {
                                'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 
                                'knee': 26, 'ankle': 28, 'foot': 32, 'ear': 8 
                              }
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0 
        
        self.FEEDBACK_ID_MAP = {
                                0: ('BEND BACKWARDS', 215, (0, 153, 255)), 
                                1: ('BEND FORWARD', 215, (0, 153, 255)),   
                                2: ('KNEE OVER TOE', 170, (255, 80, 80)),
                                3: ('SQUAT TOO DEEP', 125, (255, 80, 80)),
                                4: ('YOUR BACK IS NOT STRAIGHT', 260, (200, 100, 0)) 
                               }
        self.reset_state()


    def reset_state(self):
        self.state_tracker = {
            'state_seq': [], 'start_inactive_time': time.perf_counter(), 
            'start_inactive_time_front': time.perf_counter(), 'INACTIVE_TIME': 0.0, 
            'INACTIVE_TIME_FRONT': 0.0,
            'DISPLAY_TEXT' : np.full((5,), False), 
            'COUNT_FRAMES' : np.zeros((5,), dtype=np.int64), 
            'LOWER_HIPS': False, 'INCORRECT_POSTURE': False,
            'prev_state': None, 'curr_state':None,
            'SQUAT_COUNT': 0, 'IMPROPER_SQUAT':0
        }
        
    def _get_state(self, knee_angle):
        knee = None        
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]: knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]: knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]: knee = 3
        return f's{knee}' if knee else None
    
    def _update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(state)
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame_flags, dict_maps, lower_hips_disp):
        if lower_hips_disp: 
            draw_text(frame, 'LOWER YOUR HIPS', pos=(30, 80), text_color=(0, 0, 0), font_scale=0.6, text_color_bg=(255, 255, 0))  
        for idx in np.where(c_frame_flags)[0]: 
            if idx in dict_maps: 
                draw_text(frame, dict_maps[idx][0], pos=(30, dict_maps[idx][1]), text_color=(255, 255, 230), font_scale=0.6, text_color_bg=dict_maps[idx][2])
        return frame

    def process(self, frame: np.array, keypoints_results):
        play_sound = None 
        frame_height, frame_width, _ = frame.shape 

        if keypoints_results.pose_landmarks:
            ps_lm_list = keypoints_results.pose_landmarks.landmark 

            nose_coord_offset = get_landmark_array(ps_lm_list, self.dict_features['nose'], frame_width, frame_height)
            left_shldr_coord = get_landmark_array(ps_lm_list, self.left_features['shoulder'], frame_width, frame_height)
            right_shldr_coord = get_landmark_array(ps_lm_list, self.right_features['shoulder'], frame_width, frame_height)
            left_foot_temp_coord = get_landmark_array(ps_lm_list, self.left_features['foot'], frame_width, frame_height)
            right_foot_temp_coord = get_landmark_array(ps_lm_list, self.right_features['foot'], frame_width, frame_height)

            dist_l_sh_hip_for_side_check = abs(left_foot_temp_coord[1] - left_shldr_coord[1])
            dist_r_sh_hip_for_side_check = abs(right_foot_temp_coord[1] - right_shldr_coord[1])

            active_side_features = None
            shldr_coord = None 
            multiplier = 1
            person_facing_right = False 

            if dist_l_sh_hip_for_side_check > dist_r_sh_hip_for_side_check: 
                active_side_features = self.left_features
                shldr_coord = left_shldr_coord
                multiplier = -1
                person_facing_right = True 
            else:
                active_side_features = self.right_features
                shldr_coord = right_shldr_coord
                multiplier = 1
                person_facing_right = False
            
            elbow_coord = get_landmark_array(ps_lm_list, active_side_features['elbow'], frame_width, frame_height)
            wrist_coord = get_landmark_array(ps_lm_list, active_side_features['wrist'], frame_width, frame_height)
            hip_coord = get_landmark_array(ps_lm_list, active_side_features['hip'], frame_width, frame_height)
            knee_coord = get_landmark_array(ps_lm_list, active_side_features['knee'], frame_width, frame_height)
            ankle_coord = get_landmark_array(ps_lm_list, active_side_features['ankle'], frame_width, frame_height)
            foot_coord = get_landmark_array(ps_lm_list, active_side_features['foot'], frame_width, frame_height) 
            ear_coord = get_landmark_array(ps_lm_list, active_side_features['ear'], frame_width, frame_height) 

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord_offset) 

            if offset_angle > self.thresholds['OFFSET_THRESH']: 
                display_inactivity = False; end_time = time.perf_counter(); self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']; self.state_tracker['start_inactive_time_front'] = end_time
                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']: self.state_tracker['SQUAT_COUNT'] = 0; self.state_tracker['IMPROPER_SQUAT'] = 0; display_inactivity = True
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1); cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)
                if self.flip_frame: frame = cv2.flip(frame, 1)
                if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0; self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                draw_text(frame, 'CAMERA NOT ALIGNED PROPERLY!!!', pos=(30, frame_height-60), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                draw_text(frame, 'OFFSET ANGLE: '+str(offset_angle), pos=(30, frame_height-30), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0; self.state_tracker['prev_state'] =  None; self.state_tracker['curr_state'] = None
            
            else: 
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0 
                self.state_tracker['start_inactive_time_front'] = time.perf_counter() 
                    
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord) 
                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord) 
                angle_ear_shoulder_hip = find_angle(ear_coord, hip_coord, shldr_coord) 
        
                cv2.ellipse(frame, hip_coord, (30, 30), angle=0, startAngle=-90, endAngle=-90 + multiplier * hip_vertical_angle, color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                draw_dotted_line(frame, hip_coord, start=hip_coord[1] - 80, end=hip_coord[1] + 20, line_color=self.COLORS['blue'])
                cv2.ellipse(frame, knee_coord, (20, 20), angle=0, startAngle=-90, endAngle=-90 - multiplier * knee_vertical_angle, color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                draw_dotted_line(frame, knee_coord, start=knee_coord[1] - 50, end=knee_coord[1] + 20, line_color=self.COLORS['blue'])

                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ear_coord, shldr_coord, self.COLORS['light_blue'], 4, lineType=self.linetype) 

                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ear_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype) 

                current_state = self._get_state(int(knee_vertical_angle)) 
                self.state_tracker['curr_state'] = current_state 
                self._update_state_sequence(current_state) 

                if current_state == 's1': 
                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT']+=1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                    elif ('s2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1) or \
                         self.state_tracker['INCORRECT_POSTURE']: 
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False 
                else: 
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True 
                        self.state_tracker['INCORRECT_POSTURE'] = True 
                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                         self.state_tracker['state_seq'].count('s2')>=1: 
                            self.state_tracker['DISPLAY_TEXT'][1] = True 
                            self.state_tracker['INCORRECT_POSTURE'] = True 
                    
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2')>=1: 
                        self.state_tracker['LOWER_HIPS'] = True
                    else: 
                        self.state_tracker['LOWER_HIPS'] = False

                    if knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]: 
                        self.state_tracker['DISPLAY_TEXT'][3] = True 
                        self.state_tracker['INCORRECT_POSTURE'] = True 
                    
                    if current_state == 's2' or current_state == 's3': 
                        knee_over_toe_offset = 5 
                        actual_person_facing_right = person_facing_right
                        if self.flip_frame:
                            actual_person_facing_right = not person_facing_right

                        if actual_person_facing_right: 
                            if knee_coord[0] > foot_coord[0] + knee_over_toe_offset: 
                                self.state_tracker['DISPLAY_TEXT'][2] = True 
                                self.state_tracker['INCORRECT_POSTURE'] = True
                        else: 
                            if knee_coord[0] < foot_coord[0] - knee_over_toe_offset: 
                                self.state_tracker['DISPLAY_TEXT'][2] = True
                                self.state_tracker['INCORRECT_POSTURE'] = True
                    
                    if current_state == 's2': 
                        if not (self.thresholds.get('BACK_EAR_SHOULDER_HIP_MIN', 160) <= angle_ear_shoulder_hip <= self.thresholds.get('BACK_EAR_SHOULDER_HIP_MAX', 180)):
                            self.state_tracker['DISPLAY_TEXT'][4] = True 
                            self.state_tracker['INCORRECT_POSTURE'] = True 
                
                display_inactivity = False
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:
                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time
                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0; self.state_tracker['IMPROPER_SQUAT'] = 0; display_inactivity = True
                else:
                    self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0
              
                shldr_text_coord_x = shldr_coord[0] + 10
                shldr_text_coord_y = shldr_coord[1] - 10 
                knee_text_coord_x = knee_coord[0] + 15
                hip_text_coord_x = hip_coord[0] + 10

                if self.flip_frame: 
                    frame = cv2.flip(frame, 1)
                    shldr_text_coord_x = frame_width - shldr_coord[0] + 10 
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1': self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1 
                frame = self._show_feedback(frame, self.state_tracker['DISPLAY_TEXT'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS']) 

                if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0
                
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype) 
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype) 
                cv2.putText(frame, f"Back: {int(angle_ear_shoulder_hip)}", (shldr_text_coord_x, shldr_text_coord_y), self.font, 0.6, self.COLORS['cyan'], 2, lineType=self.linetype)
                 
                draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False 
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state 
        
        else: 
            if self.flip_frame: frame = cv2.flip(frame, 1)
            end_time = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']; display_inactivity = False
            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']: self.state_tracker['SQUAT_COUNT'] = 0; self.state_tracker['IMPROPER_SQUAT'] = 0; display_inactivity = True
            self.state_tracker['start_inactive_time'] = end_time
            draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
            draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
            if display_inactivity: play_sound = 'reset_counters'; self.state_tracker['start_inactive_time'] = time.perf_counter(); self.state_tracker['INACTIVE_TIME'] = 0.0
            self.state_tracker['prev_state'] =  None; self.state_tracker['curr_state'] = None; self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0; self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False); self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64); self.state_tracker['start_inactive_time_front'] = time.perf_counter()
            
        return frame, play_sound