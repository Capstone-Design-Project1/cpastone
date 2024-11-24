from typing import Optional
import cv2
from ultralytics import YOLO
from Image import Image
from PipedProcess import PipedProcess

class YoloProcess(PipedProcess):
    def __init__(self, model_path: str, threshold: float = 0.4):
        super().__init__()
        self.model_path = model_path
        self.threshold = threshold
        self.model: Optional[YOLO] = None

    def init(self):
        self.model = YOLO(self.model_path)

    def process(self, input_data: Image) -> tuple[Image, Optional[tuple[int, ...]]]:
        result = self.model.predict(input_data, stream=True, classes=[0], verbose=False)
        coords: Optional[tuple[int, ...]] = None
        for r in result:
            for box in r.boxes:
                conf = box.conf[0]
                if conf >= self.threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(input_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    coords = (x1, y1, x2, y2)
        return input_data, coords
