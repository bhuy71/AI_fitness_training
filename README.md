# AI_fitness_training

How to run this code:

1.Clone this repo to your local.

1.You run: pip install -r requirements.txt

### Mediapipe:

    In order to run Mediapipe code, you only need run : streamlit run UI.py
    
    Then the interface will appear and you can use with camera in real time or upload your videos and download the processed videos if you want.
### HRnet
    After cloning this repo, create a virtual environment using python 3.8.
        py -3.8 -m venv venv
        venv\Scripts\activate
    Install required packages:
        cd AI_fitness_training
        pip install -r hrnet_requirements.txt
    Run the Script:
        python hrnet_video_estimation.py pushup "C:\path\to\pushup_video.mp4"
    or for squat detection:
        python hrnet_video_estimation.py squat "C:\path\to\squat_video.mp4"
    AI_fitness_training/
    Directory Requirements
        ├── hrnet_video_estimation.py
        ├── HRNET/
        │   ├── _pycache_
        │   ├── yolov6
        │   ├── __init__.py
        │   ├── HRNET.py
        │   ├── utils.py
        ├── models/
        │   ├── hrnet_pose.onnx
        │   ├── yolov5s6.onnx
        │   └── pushup_squat_lstm_model.h5
        ├── hrnet_requirements.txt
    Link to download models folder: "https://drive.google.com/drive/folders/1PbrVLusCza-Yt7T1EB2NkJVJmghsdw4L?usp=sharing"
