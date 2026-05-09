import cv2
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path: str = 'yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.cap = None
    
    def estimate(self, frame):
        return self.model(frame, verbose=False, stream=True)
    
    def extract_keypoints(self, results):
        for r in results:
            if r.keypoints is not None:
                return r.keypoints.data
        return None

    def draw(self, results):
        for r in results:
            return r.plot()
        return None


def main():
    pose_estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = list(pose_estimator.estimate(frame))
        extract_keypoints = pose_estimator.extract_keypoints(results)
        plot = pose_estimator.draw(results)
        
        if plot is not None:
            cv2.imshow("YOLO Pose Estimation", plot)
        else:
            cv2.imshow("YOLO Pose Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
main()
