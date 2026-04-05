"""LLM provider wrapper for openAI API."""

import asyncio

import openai*
from openai types.chat import Completions

from app.config import settings
