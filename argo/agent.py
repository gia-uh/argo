import abc
import inspect

from pydantic import BaseModel
from .llm import LLM, Message


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
    async def execute(self, llm: LLM, messages: list[Message], agent: "Agent"):
        pass


class _MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def execute(self, llm: LLM, messages: list[Message], agent: "Agent"):
        return await self._target(llm, messages)


class Parameter(BaseModel):
    name: str
    description: str
    type: str
    required: bool = True


class Tool:
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
    def parameters(self) -> list[Parameter]:
        pass

    @abc.abstractmethod
    def run(self, **kwargs):
        pass


class _MethodTool(Tool):
    def __init__(self, name, description, target):
        super().__init__(name, description)
        self._target = target

    def parameters(self):
        args = inspect.get_annotations(self._target)
        return [
            Parameter(name=name, description="", type=type.__name__)
            for name, type in args.items()
        ]


DEFAULT_SYSTEM_PROMPT = """
You are {name}.

This is your description:
{description}
"""


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        *,
        skill_selector=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        self._name = name
        self._description = description
        self._llm = llm
        self._skills = []
        self._tools = []

        if skill_selector is None:
            from .utils import default_skill_selector

            skill_selector = default_skill_selector

        self._skill_selector = skill_selector
        self._system_prompt = system_prompt.format(name=name, description=description)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    async def perform(self, messages: list[Message]) -> Message:
        messages = [Message.system(self._system_prompt)] + messages
        skill: Skill = await self._skill_selector(self, self._skills, messages)
        response = await skill.execute(self._llm, messages, self)
        return Message.assistant(response)

    def add_skill(self, skill: Skill):
        self._skills.append(skill)

    def register_tool(self, tool: Tool):
        self._tools.append(tool)

    def skill(self, target):
        name = target.__name__
        description = inspect.getdoc(target)
        skill = _MethodSkill(name, description, target)
        self.add_skill(skill)
        return skill

    def tool(self, target):
        name = target.__name__
        description = inspect.getdoc(target)
        tool = _MethodTool(name, description, target)
        self.register_tool(tool)
        return tool
