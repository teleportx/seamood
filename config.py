from os import environ

from dotenv import load_dotenv

load_dotenv()


class ML:
    models_load = environ.get('ML_MODELS_LOAD')
    use_gpu = environ.get('USE_GPU') == 'TRUE'
