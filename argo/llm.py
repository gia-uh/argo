import os
import functools
import inspect
from typing import Any, Callable, Literal

import rich
import openai
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: Any

    @classmethod
    def system(cls, content: Any) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Any) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: Any) -> "Message":
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, content: Any) -> "Message":
        return cls(role="tool", content=content)

    def dump(self):
        return dict(
            role=self.role,
            content=(
                self.content.model_dump_json()
                if isinstance(self.content, BaseModel)
                else str(self.content)
            ),
        )

    def unpack[T: BaseModel](self, t: type[T]) -> T:
        if isinstance(self.content, t):
            return self.content

        if isinstance(self.content, str):
            return t.model_validate_json(self.content)

        raise TypeError(f"Cannot unpack {self.content} into {t}")


class LLM:
    def __init__(
        self,
        model: str,
        callback: Callable[[str], None] | None = None,
        verbose: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.verbose = verbose

        if base_url is None:
            base_url = os.getenv("BASE_URL")
        if api_key is None:
            api_key = os.getenv("API_KEY")

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.callback = callback

    async def complete(self, prompt: str, **kwargs) -> str:
        """Low-level method for one-shot completion with the LLM."""
        result = []

        async for chunk in await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs,
        ):
            content = chunk.choices[0].text

            if content is None:
                continue

            if self.callback:
                if inspect.iscoroutinefunction(self.callback):
                    await self.callback(content)
                else:
                    self.callback(content)

            result.append(content)

        return "".join(result)

    async def chat(self, messages: list[Message], **kwargs) -> Message:
        """Invoke chat completion on the LLM and return the assistant message."""
        result = []

        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=[message.dump() for message in messages], # type: ignore
            stream=True,
            **kwargs,
        ): # type: ignore
            content = chunk.choices[0].delta.content

            if content is None:
                continue

            if self.callback:
                if inspect.iscoroutinefunction(self.callback):
                    await self.callback(content)
                else:
                    self.callback(content)

            result.append(content)

        return Message.assistant("".join(result))

    async def create[T: BaseModel](
        self, model: type[T], messages: list[Message], **kwargs
    ) -> T:
        """
        Invoke chat completion on the LLM and parse the response into a Pydantic model.
        """
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[message.dump() for message in messages], # type: ignore
            response_format=model,
            **kwargs,
        )

        result = response.choices[0].message.parsed

        if self.verbose:
            rich.print(result)

        if result is None:
            raise ValueError("Failed to parse the response.")

        return result

    def wrap(self, target):
        llm_param = None
        parameters = inspect.signature(target).parameters

        for name, param in parameters.items():
            if issubclass(param.annotation, LLM):
                llm_param = name
                break

        if llm_param is None:
            raise TypeError("LLM parameter not found.")

        @functools.wraps(target)
        async def wrapper(*args, **kwargs):
            kwargs[llm_param] = self
            return await target(*args, **kwargs)

        # Remove the LLM param from wrapper so future
        # introspection doesn't see this parameter
        wrapper.__annotations__.pop(llm_param)

        return wrapper
