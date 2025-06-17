import os
import cv2
import asyncio
# import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from dotenv import load_dotenv
from utils.logger import logger
from transformers import CLIPProcessor
from fastapi import UploadFile, HTTPException
from Interface.embedding_service_interface import IEmbeddingService
from Service.model_manager import model_manager


load_dotenv()

onnxUrl = os.getenv("ONNX_MODEL_URL")
# token = os.getenv("Hugging_Face_Authorization_Token")

parent_dir = Path(__file__).resolve().parent.parent

path_onnx = parent_dir / "onnx" / "visual.onnx"

model_path=str(path_onnx)


class EmbeddingService(IEmbeddingService):

    def __init__(self, num_frames = None):
        
        self.num_frames = int(num_frames or os.getenv("NUMBER_OF_FRAMES"))

        # Initialize onnx
        # providers = ['DmlExecutionProvider']

        self.session = None

        # Initialize clip
        self.processor = None


    def extract_frames(self, videoPath: str) -> list:
        try:

            cap = cv2.VideoCapture(videoPath)
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if totalFrames < self.num_frames:
                frameIds = list(range(totalFrames))
            else:
                step = max(totalFrames // self.num_frames, 1)
                frameIds = [i * step for i in range(self.num_frames)]


            frames = []

            for frameId in frameIds:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)

                success, frame = cap.read()

                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frames.append(Image.fromarray(frame))

                    print(f"✅successfully read frame {frameId} from {videoPath}")

                else:
                    print(f"❌Failed to read frame {frameId} from {videoPath}")
                    continue

            cap.release()

            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames from video {videoPath}: {str(e)}")
            raise ValueError(f"Failed to extract frames from video: {videoPath}") from e
    

    def embed_video_scene(self, video_path: str) -> list:
        
        try:

            if not os.path.exists(model_path):
                raise HTTPException(500, detail=f"ONNX model not found at {model_path}")

            if self.processor is None:
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            if self.session is None:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError(f"No frames extracted from video: {video_path}")

            inputs = self.processor(images=frames, return_tensors="pt", padding=True)["pixel_values"].numpy().astype(np.float16)

            outputs = self.session.run(None, {"input": inputs})

            embedding = outputs[0]

            embedding /= np.linalg.norm(embedding, axis=-1, keepdims=True)
            
            embedding = embedding.mean(axis=0)

            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding video scene {video_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"EmbeddingError: {str(e)}")
