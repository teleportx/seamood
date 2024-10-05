"""Module for working with censorship."""

import os

from check_swear import SwearingCheck

from nudenet import NudeDetector


class ImageCensor:
    """Class made for working with image censor."""

    def __init__(self, use_gpu: bool = False):
        """Init func of an ImageCensor class."""
        self.detector = NudeDetector()
        self.device = "cuda" if use_gpu else "cpu"
        self.CENSORED_LABELS = [
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
        ]

    def is_image_ok(self, image_bytes: bytes, threshold: float = 0.4) -> bool:
        """Check if image okay.

        :param image_bytes: bytes of the image.
        :param threshold: confidence threshold. More = stricter.
        :return: True if image is okay else False.
        """
        detection_output = self.detector.detect(image_bytes)

        if detection_output:
            for current_dict in detection_output:
                if current_dict["class"] in self.CENSORED_LABELS:
                    if current_dict["score"] >= threshold:
                        return False
        return True

    def censor_image_from_path(
        self,
        image_path: str,
        delete_output: bool = True,
        output_path: str = "censored_image.jpg",
    ) -> bytes:
        """Censor image from image path.

        :param image_path: Path to the image.
        :param delete_output: True - if you do not need a file of censored image.
        :param output_path: You can specify path to the censored image.
        :return: Bytes of the censored image.
        """
        self.detector.censor(
            image_path, classes=self.CENSORED_LABELS, output_path=output_path
        )

        with open(output_path, "rb") as output_file:
            output_bytes = output_file.read()
        if delete_output:
            os.remove(path=output_path)

        return output_bytes

    def censor_image(
        self,
        image_bytes: bytes,
        delete_output: bool = True,
        output_path: str = "censored_image.jpg",
    ) -> bytes:
        """Censor image from bytes.

        :param image_bytes: Bytes of the image.
        :param delete_output: True - if you don't need a file of censored image.
        :param output_path: You can specify path to output file.
        :return: Bytes of the output image.
        """
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return self.censor_image_from_path(
            image_path=output_path, delete_output=delete_output, output_path=output_path
        )


class TextCensor:
    """Class made for working with text censor."""

    def __init__(self, censor_char: str = "*", use_gpu: bool = True):
        """Init func of a TextCensor class.

        :param censor_char: Char that will be used in replacing uncensored words.
        :param use_gpu: 'True' if you need to use GPU.
        """
        self.censor_char = censor_char
        self.text_censor = SwearingCheck()
        self.device = "cuda" if use_gpu else "cpu"

    def censor_text(
        self,
        text: str,
        threshold: float = 0.6,
        to_str: bool = True,
        return_ban_words: bool = False,
    ) -> str | list | tuple[list | str, set]:
        """Get censored text.

        :param text: Text that will be censored.
        :param threshold: Confidence threshold. Less = stricter.
        :param to_str: True - if you need a string as output.
        :param return_ban_words: True - if you need to get banned words.
        :return: Censored text.
        """
        listed_text = text.split()
        model_output = self.text_censor.predict_proba(listed_text)
        answer_text = []
        banned_words = set()

        for current_word, current_prob in zip(listed_text, model_output):
            if current_prob <= threshold:
                answer_text.append(current_word)
            else:
                answer_text.append(self.censor_char * len(current_word))
                banned_words.add(current_word)

        text_output = answer_text
        if to_str:
            text_output = " ".join(answer_text)

        if return_ban_words:
            return text_output, banned_words
        return text_output

    def is_text_ok(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text is ok.

        :param text: Text for checking.
        :param threshold: Confidence threshold. More = stricter.
        :return: True - if everything is ok.
        """
        probability = self.text_censor.predict_proba(text)
        if probability[0] <= threshold:
            return True
        return False
