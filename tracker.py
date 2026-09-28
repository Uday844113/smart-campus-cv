from ultralytics import YOLO


class PersonTracker:

    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.40,
            tracker="botsort_reid.yaml",
            verbose=False
        )

        return results[0]