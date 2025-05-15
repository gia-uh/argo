import abc
import functools
import inspect

from typing import AsyncIterator

from .llm import LLM, Message
from .prompts import DEFAULT_SYSTEM_PROMPT
from .skills import Skill, MethodSkill
from .tools import Tool, MethodTool


class AgentBase[In, Out](abc.ABC):
    @abc.abstractmethod
    async def perform(self, input: Message[In]) -> AsyncIterator[Message[Out]]:
        pass


class Agent[In, Out](AgentBase[In, Out]):
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
    def llm(self):
        return self._llm

    async def perform(self, input: Message[In]) -> AsyncIterator[Message[Out]]:
        from .context import Context
        """Main entrypoint for the agent.

        This method will select the right skill to perform the task and then execute it.
        The skill is selected based on the messages and the skills available to the agent.
        """
        context = Context(self, list(self._conversation) + [input])
        skill = await context.engage()

        messages = []

        async for m in await skill.execute(context):
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
        description = inspect.getdoc(target)
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
