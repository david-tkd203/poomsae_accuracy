import cv2
import os

class VideoRecorder:
    def __init__(self, output_dir, resolution=(1280,720), fps=30):
        self.output_dir = output_dir
        self.resolution = resolution
        self.fps = fps
        self.writers = {}
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def start_recording(self, cam_name):
        filename = os.path.join(self.output_dir, f"{cam_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writers[cam_name] = cv2.VideoWriter(filename, fourcc, self.fps, self.resolution)

    def write_frame(self, cam_name, frame):
        if cam_name in self.writers:
            self.writers[cam_name].write(frame)

    def stop_recording(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
