import sys
import os
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath('../'))

from fastapi import FastAPI

import config
import handlers
from ml_models import available_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.ML.models_load == ['*']:
        config.ML.models_load = list(available_models.keys())

    for model_name in config.ML.models_load:
        if model_name not in available_models:
            raise ValueError(f'Invalid model name: {model_name}')

        available_models[model_name].activate()

    yield

app = FastAPI(lifespan=lifespan)
app.include_router(handlers.router)
