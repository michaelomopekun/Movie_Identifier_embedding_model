from abc import ABC, abstractmethod
from fastapi import UploadFile

class IEmbeddingService(ABC):

    @abstractmethod
    def extract_frames(self, videoPath: str) -> list:
        pass

    @abstractmethod
    def embed_video_scene(self, video_path: str) -> list:
        pass