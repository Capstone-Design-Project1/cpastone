from typing import Optional

import cv2

from Image import Image
from PipedProcess import PipedProcess


class CameraProcess(PipedProcess):
    def __init__(self, camera_id: int):
        super().__init__()
        self.camera_id = camera_id
        self.camera: Optional[cv2.VideoCapture] = None

    def init(self):
        self.camera = cv2.VideoCapture(self.camera_id)

    def process(self, input_data: None) -> Image:
        ret, frame = self.camera.read()
        return frame
