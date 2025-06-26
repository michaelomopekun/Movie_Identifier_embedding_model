import os
import gc
import cv2
import math
import ffmpeg
import traceback
import subprocess
import numpy as np
from time import time
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

path_onnx = Path(os.getenv("ONNX_MODEL_prod_PATH", parent_dir / "onnx" / "visual.onnx"))

# Load processor & model
logger.info("========⌚Loading CLIP processor and ONNX session once...========")

CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.intra_op_num_threads = int(os.getenv("ONNX_INTRA_OP_NUM_THREADS"))
ONNX_SESSION = ort.InferenceSession(str(path_onnx), sess_options=SESSION_OPTIONS, providers=["CPUExecutionProvider"])

temp_dir = Path(os.getenv("TEMP_DIR_PATH", parent_dir / "temp"))

os.makedirs(temp_dir, exist_ok=True)

model_path=str(path_onnx)


class EmbeddingService(IEmbeddingService):

    def __init__(self, num_frames = None):
        
        self.num_frames = int(num_frames or os.getenv("NUMBER_OF_FRAMES"))

        # Initialize onnx
        # providers = ['DmlExecutionProvider']
        self.start_time = None

        self.session = None

        # Initialize clip
        self.processor = None


    def extract_frames(self, videoPath: str, batch_size: int = 2):
        try:
            
            logger.info(f"========⌚Initializing batched frame extraction from video: {videoPath}========")

            cap = cv2.VideoCapture(videoPath)
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if totalFrames < self.num_frames:
                frameIds = list(range(totalFrames))
            else:
                step = max(totalFrames // self.num_frames, 1)
                frameIds = [i * step for i in range(self.num_frames)]

            batch = []

            for idx, frameId in enumerate(frameIds):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
                success, frame = cap.read()

                if success:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)

                    batch.append(pil_img)
                    logger.info(f"=========✅ Successfully read frame {frameId} from {videoPath}========")
                    end_time = time()
                    elapsed_time = end_time - self.start_time if self.start_time else 0
                    logger.info(f"========⌚ Frame {frameId} extracted in {elapsed_time:.2f} seconds========")
                else:
                    logger.error(f"========❌ Failed to read frame {frameId} from {videoPath}========")
                    continue

                if len(batch) == batch_size or idx == len(frameIds) - 1:
                    logger.info(f"=========📦 Yielding batch of {len(batch)} frames========")
                    yield batch
                    del batch
                    gc.collect()
                    batch = []

            logger.info(f"========✅ Completed batched extraction from video: {videoPath}========")
            cap.release()

        except Exception as e:
            logger.error(f"========❌ Error extracting frames from video {videoPath}: {str(e)}========")
            raise ValueError(f"Failed to extract frames from video: {videoPath}") from e

    

    def embed_video_scene(self, video_path: str) -> list:
        try:
            self.start_time = time()

            if self._is_av1(video_path):
                logger.info(f"========⌚Video {video_path} is encoded with AV1 codec, converting to H264...========")
                video_path = self._convert_av1_to_h264(video_path)
                logger.info(f"========✅Converted to H264: {video_path}========")

            logger.info(f"========⌚Initializing video scene embedding: {video_path}========")

            if not os.path.exists(model_path):
                logger.error(f"❌ ONNX model not found at {model_path}")
                raise HTTPException(500, detail=f"ONNX model not found at {model_path}")

            if self.processor is None:
                logger.info("⌚ Loading CLIPImageProcessor...")
                self.processor = CLIP_PROCESSOR

            if self.session is None:
                logger.info(f"⌚ Loading ONNX model from {model_path}")
                self.session = ONNX_SESSION

            logger.info(f"✅ ONNX model inputs: {[inp.name for inp in self.session.get_inputs()]}")
            logger.info(f"✅ ONNX model outputs: {[out.name for out in self.session.get_outputs()]}")

            batch_size = int(os.getenv("BATCH_SIZE", 30))
            if batch_size <= 0:
                raise ValueError("BATCH_SIZE must be greater than 0")
            batch_size = min(batch_size, 50)

            logger.info(f"📦 Extracting and processing frames in batches of {batch_size}...")

            all_embeddings = []

            for batch in self.extract_frames(video_path, batch_size):
                inputs = self.processor(images=batch, return_tensors="np", padding=True)["pixel_values"].astype(np.float16)
                outputs = self.session.run(None, {"input": inputs})
                batch_embeddings = outputs[0]
                batch_embeddings /= np.linalg.norm(batch_embeddings, axis=-1, keepdims=True)
                all_embeddings.append(batch_embeddings)

                # Memory cleanup
                del batch, inputs, outputs, batch_embeddings
                gc.collect()

            if not all_embeddings:
                logger.error("❌ No embeddings generated from video.")
                raise HTTPException(404, detail="No embedding found for the video.")

            all_embeddings = np.vstack(all_embeddings)
            gc.collect()
            embedding = all_embeddings.mean(axis=0)

            logger.info(f"✅ Successfully embedded video scene: {video_path} with shape {embedding.shape}")

            end_time = time()
            elapsed_time = end_time - self.start_time

            logger.info(f"=========================================================================================")
            logger.info(f"==============⌚ Embedding process completed in {elapsed_time:.2f} seconds.==============")
            logger.info(f"=========================================================================================")

            return embedding

        except Exception as e:
            logger.error(f"❌ Error embedding video {video_path}: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

        

    def _is_av1(self, video_path: str) -> bool:

        logger.info(f"========⌚Checking if video {video_path} is encoded with AV1 codec========")
        try:
            probe = ffmpeg.probe(video_path)
            for stream in probe['streams']:
                if stream['codec_type'] == 'video' and stream['codec_name'].lower() == 'av1':
                    return True
            return False
        
        except Exception as e:
            logger.warning(f"========❌Could not detect codec for {video_path}: {e}========")
            return False


    def _convert_av1_to_h264(self, video_path: str) -> str:

        logger.info(f"========⌚Converting AV1 video {video_path} to H264 format========")
        try:
            output_path = os.path.join(temp_dir, "converted.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy", output_path
            ], check=True)
            return output_path
        
        except Exception as e:
            logger.error(f"========❌Error converting AV1 video {video_path} to H264: {str(e)}========")
            raise HTTPException(status_code=500, detail=f"Failed to convert AV1 video: {str(e)}")
