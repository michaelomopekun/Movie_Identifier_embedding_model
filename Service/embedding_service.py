import os
import cv2
import traceback
# import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from dotenv import load_dotenv
from utils.logger import logger
from transformers import CLIPImageProcessor
from fastapi import HTTPException
from Interface.embedding_service_interface import IEmbeddingService


load_dotenv()

onnxUrl = os.getenv("ONNX_MODEL_URL")
# token = os.getenv("Hugging_Face_Authorization_Token")

parent_dir = Path(__file__).resolve().parent.parent

path_onnx = Path("/app/onnx/visual.onnx")
# path_onnx = parent_dir / "onnx" / "visual.onnx"

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

            logger.info(f"========⌚initializing frame extraction from video: {videoPath}========")

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

                    logger.info(f"=========✅Successfully read frame {frameId} from {videoPath}========")

                else:
                    logger.error(f"========❌Failed to read frame {frameId} from {videoPath}========")
                    continue

            logger.info(f"========✅Successfully extracted {len(frames)} frames from video: {videoPath}========")

            cap.release()

            return frames
        
        except Exception as e:
            logger.error(f"========❌Error extracting frames from video {videoPath}: {str(e)}========")
            raise ValueError(f"Failed to extract frames from video: {videoPath}") from e
    

    def embed_video_scene(self, video_path: str) -> list:
        
        try:

            logger.info(f"========⌚initializing video scene embedding : {video_path}========")

            if not os.path.exists(model_path):
                logger.error(f"========❌ONNX model not found at {model_path}========")
                raise HTTPException(500, detail=f"ONNX model not found at {model_path}")

            if self.processor is None:
                logger.info(f"========⌚Loading CLIPImageProcessor from Hugging Face========")
                self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

            if self.processor is None:
                logger.error("========❌Failed to initialize CLIPImageProcessor, ensure the model is downloaded correctly.========")
                raise HTTPException(500, detail="Failed to initialize CLIPImageProcessor, ensure the model is downloaded correctly.")

            if self.session is None:
                logger.info(f"========⌚Loading ONNX model from {model_path}========")

                if not os.path.exists(model_path):
                    logger.error(f"========❌ONNX model not found at {model_path}========")
                    raise HTTPException(500, detail=f"ONNX model not found at {model_path}")
                
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            logger.info(f"========✅ONNX model inputs: {[inp.name for inp in self.session.get_inputs()]}========")
            logger.info(f"========✅ONNX model outputs: {[out.name for out in self.session.get_outputs()]}========")

            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError(f"No frames extracted from video: {video_path}")

            logger.info(f"========⌚commencing embedding video scene: {video_path} with {len(frames)} frames========")

            inputs = self.processor(images=frames, return_tensors="np", padding=True)["pixel_values"].astype(np.float16)

            outputs = self.session.run(None, {"input": inputs})

            embedding = outputs[0]

            embedding /= np.linalg.norm(embedding, axis=-1, keepdims=True)
            
            embedding = embedding.mean(axis=0)

            logger.info(f"=======✅Successfully embedded video scene: {video_path} with shape {embedding.shape}========")
            
            if embedding.size == 0:
                logger.error(f"Embedding for video {video_path} is empty.")
                raise HTTPException(404, detail="No embedding found for the video.")

            return embedding
        
        except Exception as e:
            logger.error(f"========❌Error embedding video scene {video_path}: {str(e)}\n{traceback.format_exc()}========")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
