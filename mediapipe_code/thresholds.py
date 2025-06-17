# thresholds.py

# Get thresholds for beginner mode
def get_thresholds_beginner():
    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (70, 95)
                           }    
    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
                    'HIP_THRESH'   : [10, 50],
                    'KNEE_THRESH'  : [50, 70, 95],

                    # Đổi tên key cho góc lưng: Tai-Vai-Hông
                    'BACK_EAR_SHOULDER_HIP_MIN': 145, 
                    'BACK_EAR_SHOULDER_HIP_MAX': 180, 

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 30.0,
                    'CNT_FRAME_THRESH' : 50
                }
    return thresholds

# Get thresholds for pro mode
def get_thresholds_pro():
    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (80, 100)
                           }    
    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
                    'HIP_THRESH'   : [15, 50],
                    'KNEE_THRESH'  : [50, 80, 100],

                    # Đổi tên key cho góc lưng: Tai-Vai-Hông
                    'BACK_EAR_SHOULDER_HIP_MIN': 150, 
                    'BACK_EAR_SHOULDER_HIP_MAX': 180,

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 30.0,
                    'CNT_FRAME_THRESH' : 50
                 }
    return thresholds
def get_thresholds_pushup():
    _ELBOW_ANGLE_THRESH = {
                            'STRAIGHT' : (160, 180), 
                            'BENDING'  : (100, 159), 
                            'BENT_LOW' : (80, 110)   
                           }    
    thresholds = {
                    'ELBOW_ANGLE': _ELBOW_ANGLE_THRESH,
                    'BACK_EAR_SHOULDER_HIP_MIN': 150, 
                    'BACK_EAR_SHOULDER_HIP_MAX': 180, 
                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 30.0,
                    'CNT_FRAME_THRESH' : 50    
                }
    return thresholds