from typing import Callable, Type, TypeVar
import rich
import json
import openai
from pydantic import BaseModel
import os


class Message(BaseModel):
    role: str
    content: str

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)


T = TypeVar("T", bound=BaseModel)
LLMCallback = Callable[[str], None]


class LLM:
    def __init__(
        self,
        model: str,
        callback: LLMCallback = None,
        verbose: bool = False,
        base_url: str = None,
        api_key: str = None,
    ):
        self.model = model
        self.verbose = verbose

        if base_url is None:
            base_url = os.getenv("BASE_URL")
        if api_key is None:
            api_key = os.getenv("API_KEY")

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.callback = callback

    async def chat(self, messages: list[Message], **kwargs) -> str:
        result = []

        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            stream=True,
            **kwargs,
        ):
            content = chunk.choices[0].delta.content

            if content is None:
                continue

            if self.callback:
                await self.callback(content)

            result.append(content)

        return "".join(result)

    async def parse(self, model: Type[T], messages: list[Message], **kwargs) -> T:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            response_format=model,
            **kwargs,
        )

        result = response.choices[0].message.parsed

        if self.verbose:
            rich.print(result)

        return result
