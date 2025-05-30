import logging
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apilab.routers.ml import router as ml_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application")
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="apilab/static"), name="static")
app.add_middleware(CorrelationIdMiddleware)
app.include_router(ml_router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("apilab/static/images/favicon.ico")


@app.get("/", response_model=str)
async def home():
    return "Este es un modelo de predicción de potabilidad de agua. Toma como entrada las características del agua y devuelve si es potable o no."
