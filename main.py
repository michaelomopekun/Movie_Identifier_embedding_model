from Controller.embedding_Controller import router
from fastapi import FastAPI
from Service.embedding_service import EmbeddingService


app = FastAPI(
    title="Movie Identifier Embedding Service API",
    description="API to extract frames from clip, then convert into an embedding.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Movie Identifier Embedding Service",
            "description": "Endpoints for extracting movie embeddings.",
        },
    ],
)


app.include_router(router)