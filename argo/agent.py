import inspect
import abc
from typing import AsyncIterator, Protocol
import runtime_generics

from .llm import LLM, Message
from .prompts import DEFAULT_SYSTEM_PROMPT
from .skills import Skill, MethodSkill
from .tools import Tool, MethodTool


class Agentic(Protocol):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def types(self) -> tuple[type, type]:
        pass

    @abc.abstractmethod
    def perform(self, input: Message) -> AsyncIterator[Message]:
        pass


@runtime_generics.runtime_generic
class AgentBase[In, Out](Agentic):
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def types(self):
        in_t, out_t = runtime_generics.get_type_arguments(self)
        return (in_t, out_t)

    async def perform(self, input: Message) -> AsyncIterator[Message]:
        in_t, _ = self.types

        data: In = input.unpack(in_t)

        async for m in self.process(data):
            yield Message.assistant(m)

    @abc.abstractmethod
    def process(self, input: In) -> AsyncIterator[Out]:
        pass


class ChatAgent(Agentic):
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        *,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        persistent:bool=True
    ):
        self._name = name
        self._description = description
        self._llm = llm
        self._skills = []
        self._tools = []
        self._system_prompt = system_prompt.format(name=name, description=description)
        self._conversation = [Message.system(self._system_prompt)]
        self._persistent = persistent

    @property
    def persistent(self):
        return self._persistent

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)

    @property
    def types(self):
        return (str, str)

    @property
    def llm(self):
        return self._llm

    async def perform(self, input: Message) -> AsyncIterator[Message]:
        from .context import Context
        """Main entrypoint for the agent.

        This method will select the right skill to perform the task and then execute it.
        The skill is selected based on the messages and the skills available to the agent.
        """
        context = Context(self, list(self._conversation) + [input])
        skill = await context.engage()

        messages = []

        async for m in skill.execute(context): # type: ignore
            yield m
            messages.append(m)

        self._conversation.extend(messages)

    def skill(self, target):
        if isinstance(target, Skill):
            self._skills.append(target)
            return target

        if not callable(target):
            raise ValueError("Skill must be a callable.")

        if not inspect.isasyncgenfunction(target):
            raise ValueError("Skill must be an async generator.")

        name = target.__name__
        description = inspect.getdoc(target) or ""
        skill = MethodSkill(name, description, target)
        self._skills.append(skill)
        return skill

    def tool(self, target):
        if isinstance(target, Tool):
            self._tools.append(target)
            return target

        name = target.__name__
        description = inspect.getdoc(target)
        tool = MethodTool(name, description, target)
        self._tools.append(tool)
        return tool
