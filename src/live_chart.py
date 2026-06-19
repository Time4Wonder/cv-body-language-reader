import time
import numpy as np
import matplotlib.pyplot as plt

EMOTIONEN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FARBEN = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db", "#95a5a6", "#9b59b6", "#1abc9c"]

class LiveChart:
    def __init__(self, history_seconds=10, update_interval=10):
        plt.ion()
        self.fig, (self.ax_emo, self.ax_mot) = plt.subplots(2, 1, figsize=(8, 5))
        self.fig.canvas.manager.set_window_title("Live Emotion Chart")

        self.history_seconds = history_seconds
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = None

        self.emotion_history = []
        self.motion_history = []
        self.timestamps = []

    def add_frame(self, emotion_probs, motion_speed, timestamp=None):
        if self.start_time is None:
            self.start_time = timestamp if timestamp is not None else time.time()
        t = (timestamp if timestamp is not None else time.time()) - self.start_time

        self.emotion_history.append(emotion_probs[:])
        self.motion_history.append(motion_speed)
        self.timestamps.append(t)
        self.frame_count += 1

        if self.frame_count % self.update_interval == 0:
            self._redraw()

    def _redraw(self):
        if not self.timestamps:
            return
        cutoff = self.timestamps[-1] - self.history_seconds
        start_idx = 0
        for i, t in enumerate(self.timestamps):
            if t >= cutoff:
                start_idx = i
                break
        t = self.timestamps[start_idx:]

        self.ax_emo.clear()
        self.ax_emo.set_ylim(0, 1)
        self.ax_emo.set_ylabel("Emotion Score")
        emo = np.array(self.emotion_history[start_idx:])
        for i in range(7):
            self.ax_emo.plot(t, emo[:, i], color=FARBEN[i], label=EMOTIONEN[i])
        self.ax_emo.legend(loc="upper right", fontsize=7)

        self.ax_mot.clear()
        self.ax_mot.set_ylim(0, 1)
        self.ax_mot.set_ylabel("Movement")
        self.ax_mot.plot(t, self.motion_history[start_idx:], color="orange")
        self.ax_mot.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
