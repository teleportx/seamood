"""Module for working with YandexAPI."""

import aiohttp
import config


async def _request(url: str, json_data: dict, headers: dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_data, headers=headers) as resp:
            return await resp.json()


class YandexGPT:
    """Class made for tasks with chatbots."""
    def __init__(
            self,
            role: str = "default",
            temperature: float = 0.6,
            stream: bool = False,
            max_tokens: int = 100,
            return_if_default: bool = True, *args, **kwargs
    ):
        """Init func of YandexGPT class.

        :param role: Role of the model.
        :param temperature: How creative model is.
        :param stream: 'True' if it will be a whole conversation.
        :param max_tokens: Max length of the answer.
        :param return_if_default: 'True' if you need to return default model answer.
        """
        self.default_answer = "К сожалению, я не могу ничего сказать об этом. Давайте сменим тему?"
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {config.ML.yandex_gpt_token}"
        }
        self.model_uri = f"gpt://{config.ML.yandex_gpt_catalogue}/yandexgpt-lite/latest"
        self.role = role
        self.history = [{  # for streaming task.
                    "role": "system",
                    "text": self.role
        }]
        self.return_default = return_if_default

    async def get_answer(self, prompt: str) -> str | dict | None:
        """Get answer to a prompt.

        :param prompt: Prompt to a model.
        :return: Answer to the prompt (dict if error).
        """
        json_prompt = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": self.stream,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens)
            },
            "messages": [
                {
                    "role": "system",
                    "text": self.role
                },
                {
                    "role": "user",
                    "text": prompt
                },
            ]
        }
        result = await _request(self.url, headers=self.headers, json_data=json_prompt)

        if "error" not in result.keys():
            result_text = result["result"]["alternatives"][0]["message"]["text"]
            if result_text != self.default_answer:
                return result_text
            elif self.return_default:
                return result_text
            return

        return result

    async def say(self, prompt: str) -> str | dict:
        """Get answer to a prompt (for chatbot task).

        :param prompt: Prompt to the model.
        :return: Answer to the prompt.
        """
        assert self.stream, "The model was created not for streaming task."
        self.history.append({"role": "user", "text": prompt})
        json_prompt = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": self.stream,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens)
            },
            "messages": self.history
        }

        result = await _request(self.url, headers=self.headers, json_data=json_prompt)

        if "error" not in result.keys():
            result_text = result["result"]["alternatives"][0]["message"]["text"]
            self.history.append({"role": "assistant", "text": result_text})
            return result_text

        return result

    async def __call__(self, prompt: str, *args, **kwargs) -> str | dict | None:
        """Get answer to the prompt.

        :prompt: Prompt to the model.
        :return: Answer to the prompt.
        """
        if not self.stream:
            return await self.get_answer(prompt=prompt)
        return await self.say(prompt=prompt)

    def clear_history(self):
        """Clear history of conversations."""
        self.history.clear()


class YaContinuer(YandexGPT):
    """Class made for fast searching."""
    def __init__(self, temperature: float = 0.2, *args, **kwargs):
        """Init func ot YaContinuer class.

        :param temperature: How creative model is.
        """
        role = """Тебе нужно дать максимально краткую, в два-три слова информацию на запрос пользователя."""
        stream = False
        max_tokens = 20
        return_if_default = False

        super().__init__(role, temperature, stream, max_tokens, return_if_default)


class YaTranslator:
    """Class made for translation task implemented via YandexAPI."""
    def __init__(self, *args, **kwargs):
        """Init func of YaTranslator class."""
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {config.ML.yandex_gpt_token}"
        }

    @staticmethod
    def __process_pred(model_response: list) -> str | list[str]:
        is_string = len(model_response) == 1
        answer = [current["text"] for current in model_response]
        if is_string:
            answer = answer[0]
        return answer

    async def translate(
            self,
            text: str | list[str],
            target_language: str = "ru",
            source_language: str = None
    ) -> str | list[str]:
        """Translate text to target language.

        :param text: Text to be translated.
        :param target_language: Langauge translate to.
        :param source_language: Language of text (can be None but for more accurately translation you need to specify).
        :return: Translated text.
        """
        is_string = isinstance(text, str)
        if is_string:
            text = [text]

        body = {
            "targetLanguageCode": target_language,
            "texts": text,
            "folderId": config.ML.yandex_gpt_catalogue,
        }
        if source_language:
            body["sourceLanguageCode"] = source_language

        result = await _request(
            url='https://translate.api.cloud.yandex.net/translate/v2/translate',
            json_data=body,
            headers=self.headers
            )

        if "translations" in result.keys():
            result = result["translations"]
            result = self.__process_pred(result)

        return result

    async def __call__(
            self,
            text: str | list[str],
            target_language: str = "ru",
            source_language: str = None,
            *args, **kwargs
    ) -> str | list[str]:
        """Translate text to target language via call func.

        :param text: Text to be translated.
        :param target_language: Langauge translate to.
        :param source_language: Language of text (can be None but for more accurately translation you need to specify).
        :return: Translated text.
        """
        return await self.translate(text=text, target_language=target_language, source_language=source_language)
