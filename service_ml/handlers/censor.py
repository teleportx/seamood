from fastapi import APIRouter, UploadFile

import config
from ml_models import available_models
from service_ml.models import TextModel, VerifierModel

router = APIRouter()


@router.post("/api/ml/image/censor/is_ok")
async def is_image_ok(image: UploadFile):
    image_bytes = await image.read()
    model = available_models['ImageCensor'].model

    return VerifierModel(value=model.is_image_ok(image_bytes=image_bytes, threshold=config.ML.image_censor_threshold))


@router.post("/api/ml/text/censor/is_ok")
async def is_text_ok(body: TextModel):
    model = available_models['TextCensor'].model
    return VerifierModel(value=model.is_text_ok(text=body.text, threshold=config.ML.text_censor_threshold))


@router.post("/api/ml/text/censor")
async def censor_text(body: TextModel):
    model = available_models['TextCensor'].model
    return TextModel(text=model.censor_text(text=body.text, threshold=config.ML.text_censor_threshold))
