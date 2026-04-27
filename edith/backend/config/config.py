"""Setup env configurations"""

from dotenv import load_dotenv
import os
from typing import Callable, Iterable

from constants.constants import *
from constants.text import SEPARATOR

load_dotenv()

def _get_collection_env(
    var_name: str,
    collection_type: Callable[[Iterable[str]], object],
) :
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Missing environment variable: {var_name}")

    items = (v.strip() for v in value.split(SEPARATOR))
    return collection_type(items)


FAST_API_TITLE: str | None = os.getenv(FAST_API_TITLE_ENV)
if FAST_API_TITLE is None:
    raise ValueError(f"Missing environment variable: {FAST_API_TITLE_ENV}")

ALLOWED_ORIGINS = _get_collection_env(ALLOWED_ORIGINS_ENV, list)
STOP_WORDS = _get_collection_env(STOP_WORDS_ENV, set)
START_WORDS = _get_collection_env(START_WORDS_ENV, set)
STATUS_WORDS = _get_collection_env(STATUS_WORDS_ENV, set)
SHUTDOWN_WORDS = _get_collection_env(SHUTDOWN_WORDS_ENV, set)
SWITCH_WORDS = _get_collection_env(SWITCH_WORDS_ENV, set)
