import os
from dotenv import load_dotenv

class OBSConfig:
    def __init__(self, env_path=".env"):
        load_dotenv(env_path)
        self.video_dir = os.getenv("VIDEO_DIR", "videos")
        self.report_dir = os.getenv("REPORT_DIR", "reports")
        self.default_resolution = os.getenv("RESOLUTION", "1280x720")
        self.default_fps = int(os.getenv("FPS", "30"))
        self.font_family = os.getenv("FONT_FAMILY", "Arial")
        self.theme = os.getenv("THEME", "light")
