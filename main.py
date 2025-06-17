from fastapi import FastAPI
from contextlib import asynccontextmanager
from Service.model_manager import model_manager
from Controller.embedding_Controller import router
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Movie Identifier Embedding Service...")

    try:
        yield
    finally:
        logger.info("Shutting down Movie Identifier Embedding Service...")
        await model_manager.cleanup_model()


# Create FastAPI application
app = FastAPI(
    title="Movie Identifier Embedding Service API",
    description="API to extract frames from clip, then convert into an embedding.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Movie Identifier Embedding Service",
            "description": "Endpoints for extracting movie embeddings.",
        },
    ],
)


#routers
app.include_router(router)
