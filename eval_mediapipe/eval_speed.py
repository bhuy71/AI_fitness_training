import cv2
import mediapipe as mp
import time
import os
import numpy as np
from tqdm import tqdm

# --- CẤU HÌNH ---
IMAGE_DIR = "val"  
WARMUP_RUNS = 10   

# --- KHỞI TẠO MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

def evaluate_mediapipe_speed(image_dir, warmup_runs):
    """
    Đánh giá tốc độ xử lý của MediaPipe Pose trên TẤT CẢ các ảnh trong thư mục.

    Args:
        image_dir (str): Đường dẫn đến thư mục chứa ảnh.
        warmup_runs (int): Số lượng ảnh đầu tiên dùng để khởi động.

    Returns:
        tuple: (average_inference_time_ms, average_fps, std_dev_time_ms) hoặc (None, None, None) nếu lỗi.
    """
    all_image_files_unsorted = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not all_image_files_unsorted:
        print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục '{image_dir}'.")
        return None, None, None

    all_image_files = sorted(all_image_files_unsorted)
    
    num_total_images = len(all_image_files)

    if num_total_images <= warmup_runs:
        print(f"Lỗi: Số lượng ảnh ({num_total_images}) không đủ cho {warmup_runs} lần khởi động và ít nhất 1 lần đo.")
        return None, None, None
        
    # Sử dụng tất cả các ảnh
    selected_image_files = all_image_files
    num_images_for_measurement = num_total_images - warmup_runs

    print(f"Bắt đầu đo tốc độ trên {num_images_for_measurement} ảnh, sau {warmup_runs} lần chạy khởi động.")
    print(f"Tổng số ảnh trong thư mục: {num_total_images}")
    print(f"Model complexity: 1 (đã đặt khi khởi tạo)")


    inference_times = []

    for i, img_path in enumerate(tqdm(selected_image_files, desc="Đo tốc độ MediaPipe")):
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Cảnh báo: Không thể đọc ảnh {img_path}. Bỏ qua.")
                if i >= warmup_runs: 
                    num_images_for_measurement -=1
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start_time = time.perf_counter()
            results = pose_detector.process(image_rgb)
            end_time = time.perf_counter()

            if i >= warmup_runs:
                inference_times.append(end_time - start_time)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
            if i >= warmup_runs:
                num_images_for_measurement -=1
            continue

    if not inference_times: 
        print("Không có thời gian xử lý nào được ghi lại (có thể tất cả ảnh sau khởi động đều lỗi).")
        return None, None, None, 0 

    avg_inference_time_sec = np.mean(inference_times)
    avg_fps = 1.0 / avg_inference_time_sec if avg_inference_time_sec > 0 else 0
    std_dev_time_sec = np.std(inference_times)

    avg_inference_time_ms = avg_inference_time_sec * 1000
    std_dev_time_ms = std_dev_time_sec * 1000
    return avg_inference_time_ms, avg_fps, std_dev_time_ms, len(inference_times)

if __name__ == "__main__":
    print("--- Đánh giá tốc độ MediaPipe Pose ---")
    
    avg_time_ms, avg_fps, std_time_ms, num_images_measured = evaluate_mediapipe_speed(IMAGE_DIR, WARMUP_RUNS)

    if avg_time_ms is not None:
        print(f"\n--- Kết quả đo tốc độ ---")
        print(f"Cấu hình MediaPipe:")
        print(f"  - Static Image Mode: True (đã đặt khi khởi tạo)")
        print(f"  - Model Complexity: 1 (đã đặt khi khởi tạo)")
        print(f"  - Min Detection Confidence: 0.3 (đã đặt khi khởi tạo)")

        print(f"\nThống kê trên {num_images_measured} ảnh (sau {WARMUP_RUNS} lần khởi động):")
        print(f"Thời gian xử lý trung bình mỗi ảnh: {avg_time_ms:.2f} ms")
        print(f"FPS (Frames Per Second) trung bình: {avg_fps:.2f}")
        print(f"Độ lệch chuẩn thời gian xử lý: {std_time_ms:.2f} ms")
    else:
        print("Không thể hoàn thành việc đánh giá tốc độ.")

    if 'pose_detector' in locals() and pose_detector is not None:
        print("\nClosing MediaPipe Pose detector...")
        pose_detector.close()