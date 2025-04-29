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
    def execute(self, llm: LLM, messages=list[Message]):
        pass


class _MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    def execute(self, llm: LLM, messages=list[Message]):
        return self._target(messages)


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


class Agent:
    def __init__(self, name: str, description: str, llm: LLM, *, skill_selector: None):
        self._name = name
        self._description = description
        self._llm = llm
        self._skills = []
        self._tools = []

        if skill_selector is None:
            from .utils import default_skill_selector

            skill_selector = default_skill_selector

        self._skill_selector = skill_selector

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def perform(self, messages: list[Message]) -> str:
        pass

    def teach(self, skill: Skill):
        self._skills.append(skill)

    def use(self, tool: Tool):
        self._tools.append(tool)

    def skill(self, target):
        name = target.__name__
        description = inspect.getdoc(target)
        skill = _MethodSkill(name, description, target)
        self.teach(skill)
        return skill

    def tool(self, target):
        name = target.__name__
        description = inspect.getdoc(target)
        tool = _MethodTool(name, description, target)
        self.use(tool)
        return tool
