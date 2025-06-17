import os
import json
import cv2
import mediapipe as mp
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- PATHS ---
GT_ANNOTATION_FILE = "eval_dataset_annotations.json"
IMAGE_DIR_FOR_EVALUATION = "val"
MEDIAPIPE_PREDICTIONS_FILE = "mediapipe_predictions_coco_format.json"

if not os.path.exists(GT_ANNOTATION_FILE):
    print(f"ERROR: Ground Truth annotation file not found: {GT_ANNOTATION_FILE}")
    exit()
if not os.path.isdir(IMAGE_DIR_FOR_EVALUATION):
    print(f"ERROR: Image directory for evaluation not found: {IMAGE_DIR_FOR_EVALUATION}")
    exit()

# --- COCO KEYPOINTS DEFINITION ---
COCO_KEYPOINTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# --- OKS SIGMAS (K VALUES) ---
OKS_SIGMAS = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62,
    1.07, 1.07, 0.87, 0.87, 0.89, 0.89
])

# --- MEDIAPIPE TO COCO MAPPING ---
MEDIAPIPE_TO_COCO_MAP = {
    mp_pose.PoseLandmark.NOSE.value: COCO_KEYPOINTS_NAMES.index("nose"),
    mp_pose.PoseLandmark.LEFT_EYE.value: COCO_KEYPOINTS_NAMES.index("left_eye"),
    mp_pose.PoseLandmark.RIGHT_EYE.value: COCO_KEYPOINTS_NAMES.index("right_eye"),
    mp_pose.PoseLandmark.LEFT_EAR.value: COCO_KEYPOINTS_NAMES.index("left_ear"),
    mp_pose.PoseLandmark.RIGHT_EAR.value: COCO_KEYPOINTS_NAMES.index("right_ear"),
    mp_pose.PoseLandmark.LEFT_SHOULDER.value: COCO_KEYPOINTS_NAMES.index("left_shoulder"),
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value: COCO_KEYPOINTS_NAMES.index("right_shoulder"),
    mp_pose.PoseLandmark.LEFT_ELBOW.value: COCO_KEYPOINTS_NAMES.index("left_elbow"),
    mp_pose.PoseLandmark.RIGHT_ELBOW.value: COCO_KEYPOINTS_NAMES.index("right_elbow"),
    mp_pose.PoseLandmark.LEFT_WRIST.value: COCO_KEYPOINTS_NAMES.index("left_wrist"),
    mp_pose.PoseLandmark.RIGHT_WRIST.value: COCO_KEYPOINTS_NAMES.index("right_wrist"),
    mp_pose.PoseLandmark.LEFT_HIP.value: COCO_KEYPOINTS_NAMES.index("left_hip"),
    mp_pose.PoseLandmark.RIGHT_HIP.value: COCO_KEYPOINTS_NAMES.index("right_hip"),
    mp_pose.PoseLandmark.LEFT_KNEE.value: COCO_KEYPOINTS_NAMES.index("left_knee"),
    mp_pose.PoseLandmark.RIGHT_KNEE.value: COCO_KEYPOINTS_NAMES.index("right_knee"),
    mp_pose.PoseLandmark.LEFT_ANKLE.value: COCO_KEYPOINTS_NAMES.index("left_ankle"),
    mp_pose.PoseLandmark.RIGHT_ANKLE.value: COCO_KEYPOINTS_NAMES.index("right_ankle"),
}
print("Paths, mappings, and K-values defined.")

def run_mediapipe_and_format_predictions(gt_coco, image_base_dir):
    mediapipe_coco_predictions = []
    processed_image_ids = []
    img_ids = gt_coco.getImgIds()
    print(f"Found {len(img_ids)} images in the GT annotation file.")

    for img_id in tqdm(img_ids, desc="Processing images with MediaPipe"):
        img_info = gt_coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(image_base_dir, img_filename)

        if not os.path.exists(img_path): continue
        image_cv = cv2.imread(img_path)
        if image_cv is None: continue

        img_height, img_width, _ = image_cv.shape
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            coco_keypoints_flat = [0.0] * (len(COCO_KEYPOINTS_NAMES) * 3)
            num_valid_mp_kps = 0
            sum_visibility = 0.0

            for mp_idx, landmark in enumerate(results.pose_landmarks.landmark):
                if mp_idx in MEDIAPIPE_TO_COCO_MAP:
                    coco_idx = MEDIAPIPE_TO_COCO_MAP[mp_idx]
                    px = landmark.x * img_width
                    py = landmark.y * img_height
                    visibility_score = landmark.visibility
                    coco_keypoints_flat[coco_idx * 3 + 0] = float(px)
                    coco_keypoints_flat[coco_idx * 3 + 1] = float(py)
                    coco_keypoints_flat[coco_idx * 3 + 2] = float(visibility_score)
                    if visibility_score > 0.05:
                        sum_visibility += visibility_score
                        num_valid_mp_kps += 1

            overall_score = (sum_visibility / num_valid_mp_kps) if num_valid_mp_kps > 0 else 0.0

            valid_xs = [coco_keypoints_flat[i*3] for i in range(len(COCO_KEYPOINTS_NAMES)) if coco_keypoints_flat[i*3+2] > 0.05 and coco_keypoints_flat[i*3] > 0]
            valid_ys = [coco_keypoints_flat[i*3+1] for i in range(len(COCO_KEYPOINTS_NAMES)) if coco_keypoints_flat[i*3+2] > 0.05 and coco_keypoints_flat[i*3+1] > 0]

            if not valid_xs or not valid_ys: continue

            min_x, max_x = min(valid_xs), max(valid_xs)
            min_y, max_y = min(valid_ys), max(valid_ys)
            pred_bbox_coco = [min_x, min_y, max_x - min_x, max_y - min_y]

            mediapipe_coco_predictions.append({
                "image_id": img_id, "category_id": 1, "keypoints": coco_keypoints_flat,
                "score": float(overall_score), "bbox": pred_bbox_coco
            })
            processed_image_ids.append(img_id)

    if mediapipe_coco_predictions:
        print(f"\nSaving MediaPipe predictions to {MEDIAPIPE_PREDICTIONS_FILE}...")
        with open(MEDIAPIPE_PREDICTIONS_FILE, "w") as f:
            json.dump(mediapipe_coco_predictions, f, indent=2)
        print(f"✅ MediaPipe predictions saved. Total predictions: {len(mediapipe_coco_predictions)}")
    else:
        print("No predictions made by MediaPipe to save.")
    return mediapipe_coco_predictions, processed_image_ids

def calculate_single_oks(gt_keypoints, dt_keypoints, gt_area, sigmas):
    if gt_area <= 0: return 0.0
    num_kps = gt_keypoints.shape[0]
    oks_sum = 0.0
    num_visible_gt_kps = 0
    for i in range(num_kps):
        gt_x, gt_y, gt_v = gt_keypoints[i]
        dt_x, dt_y, _ = dt_keypoints[i]
        if gt_v > 0:
            num_visible_gt_kps += 1
            dx = dt_x - gt_x
            dy = dt_y - gt_y
            denominator = 2 * gt_area * (sigmas[i]**2) + np.finfo(float).eps
            e = (dx**2 + dy**2) / denominator
            oks_sum += np.exp(-e)
    if num_visible_gt_kps == 0: return 0.0
    return oks_sum / num_visible_gt_kps

def calculate_and_print_custom_metrics(gt_coco, dt_predictions_list, sigmas_for_oks, pck_alpha=0.2):
    all_best_oks_for_gts = []
    total_correct_pck_keypoints = 0
    total_visible_gt_keypoints_for_pck = 0
    dt_by_img_id = {}
    for dt in dt_predictions_list:
        img_id = dt['image_id']
        if img_id not in dt_by_img_id: dt_by_img_id[img_id] = []
        dt_by_img_id[img_id].append(dt)

    person_cat_id = 1 
    if 'categories' in gt_coco.dataset:
        for cat in gt_coco.dataset['categories']:
            if cat['name'] == 'person':
                person_cat_id = cat['id']
                break
    
    gt_ann_ids = gt_coco.getAnnIds(catIds=[person_cat_id])
    gt_annotations = gt_coco.loadAnns(gt_ann_ids)

    for gt_ann in tqdm(gt_annotations, desc="Calculating Custom OKS & PCK"):
        img_id = gt_ann['image_id']
        gt_kps = np.array(gt_ann['keypoints']).reshape(-1, 3)
        gt_bbox = gt_ann['bbox']
        gt_area = gt_bbox[2] * gt_bbox[3]
        if gt_area <= 0: continue

        pck_ref_length = gt_bbox[3]
        if pck_ref_length <= 0: continue
        pck_threshold_dist_sq = (pck_alpha * pck_ref_length)**2

        detections_in_img = dt_by_img_id.get(img_id, [])
        best_oks_for_this_gt = 0.0
        best_dt_kps_for_pck = None

        if not detections_in_img:
            for i in range(gt_kps.shape[0]):
                if gt_kps[i, 2] > 0: total_visible_gt_keypoints_for_pck += 1
            continue

        for dt_ann in detections_in_img:
            dt_kps = np.array(dt_ann['keypoints']).reshape(-1, 3)
            current_oks = calculate_single_oks(gt_kps, dt_kps, gt_area, sigmas_for_oks)
            if current_oks > best_oks_for_this_gt:
                best_oks_for_this_gt = current_oks
                best_dt_kps_for_pck = dt_kps
        all_best_oks_for_gts.append(best_oks_for_this_gt)

        num_kps_gt = gt_kps.shape[0]
        if best_dt_kps_for_pck is not None and best_dt_kps_for_pck.shape[0] == num_kps_gt:
            for i in range(num_kps_gt):
                if gt_kps[i, 2] > 0:
                    total_visible_gt_keypoints_for_pck += 1
                    gt_x, gt_y, _ = gt_kps[i]
                    dt_x, dt_y, _ = best_dt_kps_for_pck[i]
                    dist_sq = (dt_x - gt_x)**2 + (dt_y - gt_y)**2
                    if dist_sq <= pck_threshold_dist_sq:
                        total_correct_pck_keypoints += 1
        else:
            for i in range(num_kps_gt):
                if gt_kps[i, 2] > 0: total_visible_gt_keypoints_for_pck += 1

    print("\n--- Custom Evaluation Metrics ---")
    if all_best_oks_for_gts:
        mean_overall_oks = np.mean(all_best_oks_for_gts)
        print(f"Average OKS (of best match per GT instance, using custom sigmas): {mean_overall_oks:.4f}")
    else:
        print("Average OKS: No valid GT-DT pairs found for OKS calculation.")
    if total_visible_gt_keypoints_for_pck > 0:
        pck_score = total_correct_pck_keypoints / total_visible_gt_keypoints_for_pck
        print(f"PCK@{pck_alpha:.2f} (ref: GT bbox height): {pck_score:.4f} ({total_correct_pck_keypoints}/{total_visible_gt_keypoints_for_pck})")
    else:
        print(f"PCK@{pck_alpha:.2f}: No visible GT keypoints found for PCK calculation.")

if __name__ == "__main__":
    coco_gt = None
    mediapipe_predictions = []
    processed_img_ids_mp = []
    try:
        if os.path.exists(GT_ANNOTATION_FILE) and os.path.isdir(IMAGE_DIR_FOR_EVALUATION):
            print("Loading Ground Truth annotations...")
            coco_gt = COCO(GT_ANNOTATION_FILE)
            print("Running MediaPipe and formatting predictions...")
            mediapipe_predictions, processed_img_ids_mp = run_mediapipe_and_format_predictions(coco_gt, IMAGE_DIR_FOR_EVALUATION)
        else:
            print("Error: GT annotation file or Image directory not found. Cannot proceed.")

        if mediapipe_predictions and coco_gt:
            print("\n--- Starting Standard COCO Evaluation (AP/AR) ---")
            coco_dt_for_eval = coco_gt.loadRes(mediapipe_predictions)
            coco_eval = COCOeval(coco_gt, coco_dt_for_eval, iouType='keypoints')
            # coco_eval.params.k_values = OKS_SIGMAS

            if processed_img_ids_mp:
                coco_eval.params.imgIds = sorted(list(set(processed_img_ids_mp)))
            elif coco_gt: 
                print("Warning: MediaPipe made no predictions. Standard COCO eval scores will likely be 0.")
                coco_eval.params.imgIds = coco_gt.getImgIds()

            if hasattr(coco_eval, 'params') and coco_eval.params.imgIds: 
                print("Running COCOeval.evaluate()...")
                coco_eval.evaluate()
                print("Running COCOeval.accumulate()...")
                coco_eval.accumulate()
                print("Running COCOeval.summarize()...")
                coco_eval.summarize()
            else:
                print("Skipping COCOeval as imgIds are not set (possibly no GT or no predictions).")


            print("\n--- Calculating Custom OKS and PCK ---")
            calculate_and_print_custom_metrics(coco_gt, mediapipe_predictions, OKS_SIGMAS, pck_alpha=0.2)
        else:
            if not os.path.exists(GT_ANNOTATION_FILE): print("GT annotation file is missing. Cannot evaluate.")
            elif not mediapipe_predictions: print("No MediaPipe predictions were generated or loaded. Cannot evaluate.")
    finally:
        if 'pose_detector' in locals() and pose_detector is not None:
            print("Closing MediaPipe Pose detector...")
            pose_detector.close()