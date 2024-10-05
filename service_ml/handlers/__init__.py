from fastapi import APIRouter

from . import censor
from . import text_action
from . import text_analysis

router = APIRouter()

router.include_router(censor.router)
router.include_router(text_action.router)
router.include_router(text_analysis.router)
