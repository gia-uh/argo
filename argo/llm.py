from typing import Type, TypeVar
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


class LLM:
    def __init__(self, model: str, base_url: str = None, api_key: str = None):
        self.model = model

        if base_url is None:
            base_url = os.getenv("BASE_URL")
        if api_key is None:
            api_key = os.getenv("API_KEY")

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self, messages: list[Message], callback=None, stream=False, **kwargs
    ):
        response = []

        async for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            stream=True,
            **kwargs,
        ):
            if callback:
                await callback(chunk)

            if stream:
                yield chunk.choices[0].delta.content
            else:
                response.append(chunk.choices[0].delta.content)

        if not stream:
            return "".join(response)

    async def parse(
        self, model: Type[T], messages: list[Message], **kwargs
    ) -> T:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            response_format=model,
            **kwargs,
        )

        return response.choices[0].message.parsed
