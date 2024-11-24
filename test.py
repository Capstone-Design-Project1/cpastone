import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# YOLO 모델 초기화
yolo_model_path = "/Users/gidaseul/Desktop/lstm/data/yolov5su.pt"
yolo_net = YOLO(yolo_model_path)

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# LSTM 모델 정의
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 낙상 여부 출력
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 관절 좌표 추출
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
    else:
        keypoints = [[0, 0, 0]] * 33  # 기본값 반환
    return np.array(keypoints).flatten()

# 특징 계산 (기본 및 추가)
def calculate_features(keypoints):
    head_coords = keypoints[0:3]
    shoulders_coords = np.mean([keypoints[11:14], keypoints[12:15]], axis=0)
    hssc = np.concatenate([head_coords, shoulders_coords])

    body_width = np.linalg.norm(keypoints[11:14] - keypoints[12:15])
    body_height = np.linalg.norm(keypoints[0:3] - keypoints[27:30])
    rwhc = body_width / (body_height + 1e-5)

    return np.concatenate([hssc, [rwhc]])

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

# 로그 기록 함수
def log_detection(frame_number, message):
    """추론 결과를 로그 파일에 저장"""
    log_file = "detection_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Frame {frame_number}: {message}\n")

# 추론 실행
def run_inference(video_path, model_weights, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    input_shape = (sequence_length, 10)  # 학습 때 사용한 입력 피처 수와 동일하게 수정
    model = build_lstm_model(input_shape)
    model.load_weights(model_weights)  # 학습된 가중치 로드

    sequence = []
    prev_keypoints = None
    current_frame = 0
    fall_detected = False  # 낙상 감지 상태 플래그

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 탐지
        results = yolo_net(frame)
        detections = results[0].boxes

        coords = None
        for box in detections:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            if class_id == 0 and confidence > 0.3:  # 사람 클래스
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coords = (x1, y1, x2, y2)
                break

        if coords:
            x1, y1, x2, y2 = coords
            cropped_frame = frame[y1:y2, x1:x2]
            if cropped_frame.size > 0:
                # MediaPipe로 관절 데이터 추출
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                keypoints = extract_keypoints(results)

                if np.any(keypoints):
                    velocity, angle, com_change = calculate_advanced_features(keypoints, prev_keypoints)
                    feature = np.concatenate([calculate_features(keypoints), [velocity, angle, com_change]])
                    sequence.append(feature)

                    if len(sequence) >= sequence_length:
                        input_data = np.expand_dims(sequence[-sequence_length:], axis=0)  # 시퀀스 생성
                        prediction = model.predict(input_data)[0][0]  # 낙상 여부 예측

                        if prediction > 0.5:
                            fall_detected = True  # 낙상 감지 플래그 설정
                            message = "Fall detected!"
                            cv2.putText(frame, "FALL DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            print(f"Frame {current_frame}: {message}")
                            log_detection(current_frame, message)  # 로그 저장
                        else:
                            message = "No fall detected."
                            print(f"Frame {current_frame}: {message}")
                            log_detection(current_frame, message)  # 로그 저장

                        sequence.pop(0)

                    prev_keypoints = keypoints
                else:
                    print(f"Frame {current_frame}: MediaPipe failed to extract keypoints.")

        # 화면에 프레임 표시 (낙상이 감지된 이후부터만)
        if fall_detected:
            cv2.imshow("Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    
# 메인 실행
if __name__ == "__main__":
    base_path = "/Users/gidaseul/Desktop/lstm"
    video_folder = os.path.join(base_path, "data/videos")
    model_weights_path = os.path.join(base_path, "data/output/lstm_weights.weights.h5")

    test_video = os.path.join(video_folder, "FD_In_H11H22H33_0004_20201231_19.mp4")  # 테스트 비디오 경로
    print("Running inference...")
    run_inference(test_video, model_weights_path)
