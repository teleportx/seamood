from fastapi import APIRouter
from starlette.responses import Response

from ml_models import available_models
from service_ml.models import ContinueTextModel, TTSModel, TranslateTextModel, TextModel, GPTModel

router = APIRouter()


@router.post("/api/ml/text/tts")
async def to_neural_speech(body: TTSModel):
    model = available_models['NeuralSpeaker'].model
    return Response(
        content=model.get_bytes(text=body.text, speed=body.speed), media_type="audio/wav"
    )


@router.post("/api/ml/text/translate")
async def translate_text(body: TranslateTextModel):
    model = available_models['Translator'].model
    return TextModel(text=await model.translate(
        text=body.text,
        source_language=body.source_lang,
        target_language=body.target_lang
    ))


@router.post("/api/ml/text/continue")
async def continue_phrase(body: ContinueTextModel):
    model = available_models["Continuer"].model
    return TextModel(text=await model(prompt=body.text))


@router.post("/api/ml/text/gpt")
async def gpt_prompt(body: GPTModel):
    model = available_models["GPT"].model
    model.stream = body.dialogue
    return TextModel(text=await model(prompt=body.text))


@router.post("/api/ml/text/getcv")
async def make_cv(body: TextModel):
    model = available_models["CVMaker"].model
    return TextModel(text=await model(prompt=body.text))
