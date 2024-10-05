from fastapi import APIRouter

import config
from ml_models import available_models
from service_ml.models import TextModel, VerifierModel

router = APIRouter()


@router.get("/api/ml/text/embedding")
async def get_text_embedding(text: str):
    model = available_models['Embedder'].model
    return model.get_text_embedding(text)


@router.post("/api/ml/text/is_positive")
async def is_text_positive(body: TextModel):
    model = available_models['SentimentAnaliz'].model
    return VerifierModel(value=model.is_positive(text=body.text))


@router.get("/api/ml/text/summarization")
async def get_text_summarization(text: str):
    model = available_models['TextSummarization'].model
    return TextModel(text=model.get_sum(text=text, threshold=config.ML.text_summarization_threshold))
