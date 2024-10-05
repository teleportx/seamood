"""The module for working with text."""

import os

import torch

import torchaudio

import transformers.modeling_outputs
from transformers import AutoTokenizer, BertModel, BertTokenizer, VitsModel, pipeline


bert_output_type = (
    transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
)
tts_output_type = transformers.models.vits.modeling_vits.VitsModelOutput

embedding_size = 768


class Embedder:
    """Class made for getting embeddings of some texts.

    Need for recommender systems.
    """

    def __init__(self, bert_mode: str = "bert-base-uncased", use_gpu: bool = False):
        """Init function of an Embedder.

        :param bert_mode: Specify which model you will use.
        :param use_gpu: 'True' if you need to use GPU.
        """
        self.device = "cuda" if use_gpu else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode)
        self.model = BertModel.from_pretrained(bert_mode).to(self.device)

    @staticmethod
    def _embedding_post_process(output_tensor: bert_output_type) -> list:
        """:param output_tensor: Output of the Bert."""
        return output_tensor.pooler_output[0].tolist()

    def get_text_embedding(self, text: str | list) -> list:
        """Get embedding of the text.

        :param text: Text that you need to get embedding
        (can be a list of texts)
        :return: Embedded text
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return self._embedding_post_process(outputs)


class SentimentAnaliz:
    """Class made for sentiment analiz (positive or negative)."""

    def __init__(self, use_gpu: bool = False, language: str = "rus"):
        """Init function of SentimentAnaliz.

        :param use_gpu: 'True' if you need to use GPU.
        :param language: Specify language of the text.
        """
        self.device = "cuda" if use_gpu else "cpu"
        self.available_languages = ("rus", "eng")
        assert (
            language in self.available_languages
        ), f"Language '{language}' is not supported"

        self.language_to_model = {
            "eng": pipeline("sentiment-analysis", device=self.device),
            "rus": pipeline(
                model="seara/rubert-tiny2-russian-sentiment", device=self.device
            ),
        }

        self.nlp_model = self.language_to_model[language]

    def is_positive(
        self, text: str | list[str], threshold: float = 0.4
    ) -> bool | list[bool]:
        """Check if text is positive ('True' if positive else 'False').

        :param text: text to be checked (can be a list of texts)
        :param threshold: confidence threshold. More = stricter.
        :return: If text is okay - True, else - False
        (list of bools if list of textx was given in input)
        """
        if isinstance(text, str):
            output = self.nlp_model(text)[0]
            if output["label"].lower() == "positive" and output["score"] >= threshold:
                return True
            return False

        if isinstance(text, list):
            outputs = self.nlp_model(text)
            answer = [
                out["label"].lower() == "positive" and out["score"] >= threshold
                for out in outputs
            ]
            return answer


class ToSpeech:
    """Class made for text-to-speech task via AI."""

    def __init__(self, language: str = "rus", use_gpu: bool = True):
        """Init func of ToSpeech class.

        :param language: Language of text.
        :param use_gpu: 'True' if you will use gpu.
        """
        self.supported_languages = ("rus", "eng")
        assert (
            language in self.supported_languages
        ), f"Supported languages: {self.supported_languages}"

        self.device = "cuda" if use_gpu else "cpu"
        self.usage_case = f"facebook/mms-tts-{language}"
        self.model = VitsModel.from_pretrained(self.usage_case).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.usage_case)
        self.default_sample_rate = 16000

    def predict(self, text: str) -> tts_output_type:
        """Get prediction from text in chosen language.

        :param text: Text to get speech.
        :return: Model's output.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output

    def get_file(self, text: str, path: str = "speech.wav", speed: float = 1.0) -> None:
        """Get file of speech in audio format.

        :param text: Text to get speech.
        :param path: Path for output file.
        :param speed: Speed of speech (Default = 1.0, from 0.1 to 2.0).
        """
        output = self.predict(text=text).waveform.cpu()
        torchaudio.save(
            uri=path, src=output, sample_rate=int(self.default_sample_rate * speed)
        )

    def get_bytes(
        self,
        text: str,
        path: str = "speech.wav",
        speed: float = 1.0,
        delete_saved_file: bool = True,
    ) -> bytes:
        """Get bytes of audio file with speech.

        :param text: Text to get speech.
        :param path: Path for output file.
        :param speed: Speed of speech (Default = 1.0, from 0.1 to 2.0).
        :param delete_saved_file: 'True' if you don't need audio file.
        :return: Bytes of audio file with speech.
        """
        self.get_file(text=text, path=path, speed=speed)
        with open(path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        if delete_saved_file:
            os.remove(path)

        return audio_bytes


class TextSummarization:
    """Class made for text summarization task."""

    def __init__(
        self,
        model_name: str = "UrukHan/t5-russian-summarization",
        use_gpu: bool = False,
    ):
        """Init func of TextSummarization class.

        :param model_name: Name of the Summarization model.
        :param use_gpu: 'True' if you need to use GPU.
        """
        self.device = "cuda" if use_gpu else "cpu"
        self.model = pipeline(model=model_name, device=self.device)

    def get_sum(self, text: str | list[str], threshold: int = 15) -> str | list[str]:
        """Get summarization of the text.

        :param text: Text to get summarization (can be a list of texts).
        :param threshold: Limit for not-summarized text
        :return: Summarization (list).
        """
        output = self.model(text)

        if isinstance(text, str):
            if len(text) > threshold:
                return output[0]["generated_text"]
            else:
                return text
        if isinstance(text, list):
            return [output[i]["generated_text"] if len(text[i]) > threshold else text[i] for i in range(len(text))]
