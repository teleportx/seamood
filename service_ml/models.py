from pydantic import BaseModel, Field


class TextModel(BaseModel):
    text: str


class ContinueTextModel(TextModel):
    with_prompt: bool


class TTSModel(TextModel):
    speed: float = Field(ge=1, le=2)


class TranslateTextModel(TextModel):
    target_lang: str
    source_lang: str = None


class VerifierModel(BaseModel):
    value: bool


class GPTModel(TextModel):
    dialogue: bool
