import inspect
import abc
from typing import AsyncIterator, Callable, Protocol

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


class AgentBase[In, Out](Agentic):
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def description(self):
        return self.__class__.__doc__ or ""

    @property
    def types(self):
        return self.__class__.__orig_bases__[0].__args__ # type: ignore

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
        persistent:bool=True,
        skills: list | None = None,
        tools: list | None = None,
        context_cls = None,
        skill_cls = None,
        tool_cls = None,
        prompt_callback: Callable | None = None,
    ):
        from .context import Context

        self._name = name
        self._description = description
        self._llm = llm
        self._skills = []
        self._tools = []
        self._system_prompt = system_prompt.format(name=name, description=description)
        self._conversation = [Message.system(self._system_prompt)]
        self._persistent = persistent
        self._skill_cls = skill_cls or MethodSkill
        self._tool_cls = tool_cls or MethodTool
        self._context_cls = context_cls or Context
        self._prompt_callback = prompt_callback

        # initialize predefined skills and tools
        for skill in skills or []:
            self.skill(skill)

        for tool in tools or []:
            self.tool(tool)

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
        """Main entrypoint for the agent.

        This method will select the right skill to perform the task and then execute it.
        The skill is selected based on the messages and the skills available to the agent.
        """
        conversation_len = len(self._conversation)
        context = self._context_cls(self, list(self._conversation) + [input])
        skill = await context.engage()
        await skill.execute(context)
        self._conversation = context.messages

        for m in self._conversation[conversation_len:]:
            yield m

    def skill(self, target) -> Skill:
        """
        Add a method as a skill to the agent.
        The method must be an async generator.
        """
        if isinstance(target, Skill):
            self._skills.append(target)
            return target

        if not callable(target):
            raise ValueError("Skill must be a callable.")

        if not inspect.iscoroutinefunction(target):
            raise ValueError("Skill must be a coroutine function.")

        name = target.__name__
        description = inspect.getdoc(target) or ""
        skill = self._skill_cls(name, description, target)
        self._skills.append(skill)
        return skill

    def tool(self, target) -> Tool:
        """
        Adds a method as a tool to the agent.

        If the method expects an LLM at any keyword parameter, the
        agent will automatically inject it.

        The method must be an async function.
        """

        if isinstance(target, Tool):
            self._tools.append(target)
            return target

        if not callable(target):
            raise ValueError("Tool must be a callable.")

        if not inspect.iscoroutinefunction(target):
            raise ValueError("Tool must be a coroutine function.")

        name = target.__name__
        description = inspect.getdoc(target) or ""
        signature = inspect.signature(target).parameters

        # If the method expects an LLM, wrap it
        # to inject the agent's LLM instance
        if any(issubclass(param.annotation, LLM) for param in signature.values()):
            target = self.llm.wrap(target)

        tool = self._tool_cls(name, description, target)
        self._tools.append(tool)
        return tool
