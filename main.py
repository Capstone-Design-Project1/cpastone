import os
import cv2
import numpy as np
import json
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# YOLO 모델 초기화 (더 일반적인 모델로 교체 가능)
yolo_model_path = "/Users/gidaseul/Desktop/lstm/data/yolov5su.pt"
yolo_net = YOLO(yolo_model_path)

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# JSON 데이터 로드 함수
def load_json_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        start_frame = int(float(data['annotations']['object'][0]['startFrame']))
        end_frame = int(float(data['annotations']['object'][0]['endFrame']))
        return start_frame, end_frame

# 관절 좌표 추출
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
    else:
        keypoints = [[0, 0, 0]] * 33  # 기본값 반환
    return np.array(keypoints).flatten()

# 기본 특징 계산
def calculate_features(keypoints):
    head_coords = keypoints[0:3]
    shoulders_coords = np.mean([keypoints[11:14], keypoints[12:15]], axis=0)
    hssc = np.concatenate([head_coords, shoulders_coords])

    body_width = np.linalg.norm(keypoints[11:14] - keypoints[12:15])
    body_height = np.linalg.norm(keypoints[0:3] - keypoints[27:30])
    rwhc = body_width / (body_height + 1e-5)

    return np.concatenate([hssc, [rwhc]])

# 추가 특징 계산
def calculate_advanced_features(keypoints, prev_keypoints=None):
    head_y = keypoints[1]
    prev_head_y = prev_keypoints[1] if prev_keypoints is not None else head_y
    velocity = head_y - prev_head_y

    shoulders_vec = keypoints[11:14] - keypoints[12:15]
    hips_vec = keypoints[23:26] - keypoints[24:27]
    dot_product = np.dot(shoulders_vec, hips_vec)
    angle = np.arccos(dot_product / (np.linalg.norm(shoulders_vec) * np.linalg.norm(hips_vec) + 1e-5))

    com = np.mean(keypoints.reshape(33, 3), axis=0)
    prev_com = np.mean(prev_keypoints.reshape(33, 3), axis=0) if prev_keypoints is not None else com
    com_change = np.linalg.norm(com - prev_com)

    return velocity, angle, com_change

# LSTM 모델 정의
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 학습 데이터 준비
def prepare_training_data(video_path, json_file, sequence_length=30):
    start_frame, end_frame = load_json_annotations(json_file)
    cap = cv2.VideoCapture(video_path)

    sequence = []
    features = []
    labels = []

    prev_keypoints = None
    current_frame = 0

    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break

        if current_frame >= start_frame:
            results = yolo_net(frame)
            detections = results[0].boxes

            coords = None
            for box in detections:
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                if class_id == 0 and confidence > 0.3:  # 신뢰도 기준 완화
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    coords = (x1, y1, x2, y2)
                    break

            if coords:
                x1, y1, x2, y2 = coords
                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.size > 0:
                    rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    keypoints = extract_keypoints(results)

                    if np.any(keypoints):
                        velocity, angle, com_change = calculate_advanced_features(keypoints, prev_keypoints)
                        feature = np.concatenate([calculate_features(keypoints), [velocity, angle, com_change]])
                        sequence.append(feature)

                        if len(sequence) >= sequence_length:
                            features.append(sequence[-sequence_length:])
                            labels.append(1)
                            sequence.pop(0)

                        prev_keypoints = keypoints
                    else:
                        print(f"Frame {current_frame}: MediaPipe failed to extract keypoints.")

        current_frame += 1

    cap.release()
    print(f"Total sequences generated: {len(features)}")
    if len(features) == 0:
        raise ValueError("No training data generated. Check input files and processing steps.")
    return np.array(features), np.array(labels)

# 학습 실행
def train_lstm(features, labels, save_path="lstm_weights.weights.h5"):
    input_shape = (features.shape[1], features.shape[2])
    model = build_lstm_model(input_shape)
    model.fit(features, labels, epochs=20, batch_size=32, validation_split=0.2)
    model.save_weights(save_path)
    print(f"Model weights saved at {save_path}")


# 메인 실행
if __name__ == "__main__":
    base_path = "/Users/gidaseul/Desktop/lstm"
    video_folder = os.path.join(base_path, "data/videos")
    json_folder = os.path.join(base_path, "data/json")
    output_model_path = os.path.join(base_path, "data/output/lstm_weights.weights.h5")

    json_file = os.path.join(json_folder, "FD_In_H11H22H32_0004_20201016_13.json")
    video_file = os.path.join(video_folder, "FD_In_H11H22H32_0004_20201016_13.mp4")

    print("Preparing training data...")
    features, labels = prepare_training_data(video_file, json_file)
    print(f"Training LSTM model with {features.shape[0]} sequences...")
    train_lstm(features, labels, save_path=output_model_path)
