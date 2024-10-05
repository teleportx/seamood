"""All available NN models."""
from http.client import HTTPException
from typing import Dict

import config
from ml_utils import censor, text, yandex


class MLModelStorer:
    def __init__(self, model_class):
        self.model_class = model_class
        self._model = None

    def activate(self):
        self._model = self.model_class(use_gpu=config.ML.use_gpu)

    @property
    def model(self):
        if self._model is None:
            raise HTTPException(501, f'Model not loaded.')
        return self._model


available_models: Dict[str, MLModelStorer] = {
    "ImageCensor": MLModelStorer(censor.ImageCensor),
    "TextCensor": MLModelStorer(censor.TextCensor),
    "Embedder": MLModelStorer(text.Embedder),
    "SentimentAnaliz": MLModelStorer(text.SentimentAnaliz),
    "NeuralSpeaker": MLModelStorer(text.ToSpeech),
    "TextSummarization": MLModelStorer(text.TextSummarization),
    "Translator": MLModelStorer(yandex.YaTranslator),
    "Continuer": MLModelStorer(yandex.YaContinuer),
    "GPT": MLModelStorer(yandex.YandexGPT),
    "CVMaker": MLModelStorer(yandex.YaCVMaker),
}
