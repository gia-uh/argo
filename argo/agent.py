import abc
import functools
import inspect

from .llm import LLM, Message
from .prompts import DEFAULT_SYSTEM_PROMPT
from .skills import Skill, MethodSkill
from .tools import Tool, MethodTool


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        *,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        self._name = name
        self._description = description
        self._llm = llm
        self._skills = []
        self._tools = []
        self._system_prompt = system_prompt.format(name=name, description=description)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def llm(self):
        return self._llm

    async def perform(self, messages: list[Message]) -> Message:
        from .context import Context
        """Main entrypoint for the agent.

        This method will select the right skill to perform the task and then execute it.
        The skill is selected based on the messages and the skills available to the agent.
        """
        context = Context(self, [Message.system(self._system_prompt)] + list(messages))
        skill = await context.engage()
        return await skill.execute(context)

    def skill(self, target):
        if isinstance(target, Skill):
            self._skills.append(target)
            return target

        if not callable(target):
            raise ValueError("Skill must be a callable.")

        if not inspect.iscoroutinefunction(target):
            raise ValueError("Skill must be a coroutine function.")

        name = target.__name__
        description = inspect.getdoc(target)
        skill = MethodSkill(name, description, target)
        self._skills.append(skill)
        return skill

    def tool(self, target):
        if isinstance(target, Tool):
            self._tools.append(target)
            return target

        # BUG: Doesn't work for sync method
        if not inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def wrapper(*args, **kwargs):
                return target(*args, **kwargs)

            target = wrapper

        name = target.__name__
        description = inspect.getdoc(target)
        tool = MethodTool(name, description, target)
        self._tools.append(tool)
        return tool
