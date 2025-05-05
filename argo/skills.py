import abc
from .llm import Message


class Skill:
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
        self._before = []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @abc.abstractmethod
    async def _execute(self, ctx) -> Message:
        pass

    async def execute(self, ctx) -> Message:
        messages = list(ctx.messages)

        for skill in self._before:
            m = await skill.execute(ctx)
            messages.append(m)

        return await self._execute(ctx)

    def requires(self, target):
        if isinstance(target, Skill):
            self._before.append(target)
        else:
            self._before.append(MethodSkill(target.__name__, target.__doc__, target))

        return target


class MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def _execute(self, ctx) -> Message:
        return await self._target(ctx)
