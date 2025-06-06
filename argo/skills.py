import abc

from .llm import Message
from typing import AsyncIterator


class Skill:
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @abc.abstractmethod
    async def execute(self, ctx) -> AsyncIterator[Message]:
        pass


class MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def execute(self, ctx): # type: ignore
        async for m in self._target(ctx):
            yield m


async def chat(ctx: "Context"):
    """
    Casual chat with the user.
    """
    yield await ctx.reply()
