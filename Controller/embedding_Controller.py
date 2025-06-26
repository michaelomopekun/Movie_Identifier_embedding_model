import os
import shutil
from time import time
from pathlib import Path
from utils.logger import logger
from Service.model_manager import model_manager
from fastapi import APIRouter, UploadFile, File, HTTPException
from model.embedding_response_model import EmbeddingResponseModel
from Service.embedding_service import EmbeddingService as _embeddingService

router = APIRouter()
embedding_service = _embeddingService()
parent_dir = Path(__file__).resolve().parent.parent


@router.post("/embeddingModel", response_model=EmbeddingResponseModel)
async def extract_embedding(file: UploadFile = File(...)):

    try:
        start_time = time()

        onnx_url = os.getenv("ONNX_MODEL_URL")

        parent = Path(os.getenv("ONNX_MODEL_prod_PATH", parent_dir / "onnx" / "visual.onnx"))

        if not parent.exists():
            logger.info("⌚========Initializing ONNX model download...========")

            if not onnx_url:
                raise ValueError("ONNX_MODEL_URL environment variable is not set.")
            
            await model_manager.download_model(onnx_url, parent)

        client_upload_temp_dir = "client_uploads"

        os.makedirs(client_upload_temp_dir, exist_ok=True)

        temp_dir =  "temp"

        os.makedirs(temp_dir, exist_ok=True)

        temp_file_path = os.path.join(temp_dir, "converted.mp4")

        client_upload_temp_file_path = os.path.join(client_upload_temp_dir, file.filename)

        with open(client_upload_temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_results = embedding_service.embed_video_scene(client_upload_temp_file_path)
        if raw_results is None or raw_results.size == 0:
            raise HTTPException(status_code=404, detail="No results found")
        
        result = raw_results

        end_time = time()
        elapsed_time = end_time - start_time

        logger.info(f"=========================================================================================")
        logger.info(f"==============⌚all processes completed in {elapsed_time:.2f} seconds.==============")
        logger.info(f"=========================================================================================")

        return {"embedding": result}

    except Exception as e:
        logger.error(f"========❌Error processing file {file.filename}: {str(e)}========")
        # model_manager.cleanup_model()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(client_upload_temp_file_path):
            os.remove(client_upload_temp_file_path)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

