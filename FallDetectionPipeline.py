from CameraProcess import CameraProcess
from MediapipeProcess import MediapipeProcess
from YoloProcess import YoloProcess
from LSTMModel import build_lstm_model
import numpy as np

def run_pipeline(video_path, lstm_weights):
    camera_process = CameraProcess(video_path)
    yolo_process = YoloProcess(model_path="yolov5_weights.pt")
    mediapipe_process = MediapipeProcess({'min_detection_confidence': 0.6, 'min_tracking_confidence': 0.6})
    
    # Pipeline 초기화
    pipeline = Pipeline(camera_process, yolo_process, mediapipe_process)
    pipeline.start()

    # LSTM 모델 로드
    model = build_lstm_model(input_shape=(30, 99))
    model.load_weights(lstm_weights)

    # 추론
    prediction = model.predict(features)
    print("Fall detected!" if prediction > 0.5 else "No fall detected.")
