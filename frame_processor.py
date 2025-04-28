import os
import cv2
import easyocr
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def is_significantly_different(img1, img2, threshold=0.9):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold


def extract_frames(video_path, fps=1, output_dir="frames", cache_path="frames.txt"):
    if os.path.exists(cache_path):
        print(f"Loading frame list from {cache_path}.")
        with open(cache_path, 'r', encoding="utf-8") as file:
            frames = file.read().splitlines()
        return frames

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    last_frame = None
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % int(frame_rate) == 0:
            if last_frame is None or is_significantly_different(last_frame, frame):
                frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                frames.append(frame_filename)
                last_frame = frame
                frame_count += 1

    cap.release()

    with open(cache_path, 'w', encoding="utf-8") as file:
        file.write("\n".join(frames))
    print(f"Frames saved to {cache_path}.")
    return frames


def ocr_frames(frame_dir="frames", cache_path="ocr_results.txt"):
    if os.path.exists(cache_path):
        print(f"Loading OCR results from {cache_path}.")
        with open(cache_path, 'r', encoding="utf-8") as file:
            return file.read().splitlines()

    reader = easyocr.Reader(['en'], gpu=True)
    ocr_results = []
    
    for frame_filename in tqdm(os.listdir(frame_dir), desc="Performing OCR"):
        if frame_filename.endswith(".jpg"):
            img_path = os.path.join(frame_dir, frame_filename)
            result = reader.readtext(img_path)
            ocr_results.append(str(result))

    with open(cache_path, 'w', encoding="utf-8") as file:
        file.write("\n".join(ocr_results))
    print(f"OCR results saved to {cache_path}.")
    return ocr_results