import os
import atexit
import asyncio
import requests
from tqdm import tqdm
from pathlib import Path
from utils.logger import logger


class ModelManager:

    def __init__(self):
        self.path_onnx = Path(__file__).resolve().parent.parent / "onnx" / "visual.onnx"
        atexit.register(self.cleanup_model)

    async def cleanup_model(self):
        try:
            if self.path_onnx.exists():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, os.remove, str(self.path_onnx))
                logger.info(f"Cleaned up ONNX model at {self.path_onnx}")

        except Exception as e:
            logger.error(f"Failed to cleanup ONNX model: {str(e)}")
            raise

    async def download_model(self, url: str, path: Path) -> None:
        try:
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0"
            }, stream=True)

            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading ONNX model from {url}")

            with open(path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
                        
            logger.info("Model download completed successfully")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise

# Create a singleton instance
model_manager = ModelManager()