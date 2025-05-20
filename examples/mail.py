import asyncio
import random

from argo.agent import AgentBase
from argo.llm import LLM
from pydantic import BaseModel
from argo.crew import Crew

import os
import dotenv

dotenv.load_dotenv()


class EmailStartup(BaseModel):
    host: str
    port: int
    username: str
    password: str


class EmailItem(BaseModel):
    sender: str
    body: str


class EmailFetcher(AgentBase[EmailStartup, EmailItem]):
    """
    Retrieves emails from a specific user and server.
    """

    async def process(self, input: EmailStartup):
        while True:
            # yield EmailItem(

            # )