from typing import Optional

import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

from Image import Image
from PipedProcess import PipedProcess


class MediapipeProcess(PipedProcess):
    def __init__(self, model_params: dict):
        super().__init__()
        self.model_params = model_params
        self.model: Optional[Pose] = None

    def init(self):
        self.model = Pose(**self.model_params)

    def process(self, input_data: tuple[Image, Optional[tuple[int, ...]]]) -> None:
        img, coords = input_data
        if coords is not None:
            x1, y1, x2, y2 = coords
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size > 0:
                rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                results = self.model.process(rgb_img)
                if results.pose_landmarks:
                    draw_landmarks(cropped_img, results.pose_landmarks, POSE_CONNECTIONS)
                    img[y1:y2, x1:x2] = cropped_img
        cv2.imshow('Result', img)
        cv2.waitKey(1)
