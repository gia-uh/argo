import abc
from .llm import Message
from typing import AsyncIterator
from pydantic import BaseModel


class Skill[T: BaseModel | str]:
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
    async def execute(self, ctx) -> AsyncIterator[Message[T]]:
        pass


class MethodSkill[T: BaseModel | str](Skill[T]):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def execute(self, ctx) -> AsyncIterator[Message[T]]:
        async for m in await self._target(ctx):
            yield m
