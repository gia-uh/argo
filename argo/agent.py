import abc
import inspect

from pydantic import BaseModel, create_model
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
    async def execute(self, agent: "Agent", messages: list[Message]) -> Message:
        pass


class _MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def execute(self, agent: "Agent", messages: list[Message]) -> Message:
        return await self._target(agent, messages)


DEFAULT_TOOL_INVOKE_PROMPT = """
Given the previous messages, your task
is to generate parameters to invoke the following tool.

Name: {name}.

Parameters:
{parameters}

Description:
{description}

Return the appropriate parameters as a JSON object
with the following format:
{format}
"""


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
    def parameters(self) -> dict[str, type]:
        pass

    @abc.abstractmethod
    async def run(self, **kwargs):
        pass

    async def invoke(self, agent: "Agent", messages: list[Message]) -> str:
        model_cls: type[BaseModel] = create_model(
            "Tool_" + self.name, **self.parameters()
        )

        prompt = DEFAULT_TOOL_INVOKE_PROMPT.format(
            name=self.name,
            parameters=self.parameters(),
            description=self.description,
            format=model_cls.model_json_schema(),
        )

        response: BaseModel = await agent.llm.parse(
            model_cls, messages + [Message.system(prompt)]
        )
        return await self.run(**response.model_dump())


class _MethodTool(Tool):
    def __init__(self, name, description, target):
        super().__init__(name, description)
        self._target = target

    def parameters(self):
        args = inspect.get_annotations(self._target)
        return {name: type for name, type in args.items() if name != "return"}

    async def run(self, **kwargs):
        return await self._target(**kwargs)


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

    @property
    def llm(self):
        return self._llm

    async def perform(self, messages: list[Message]) -> Message:
        messages = [Message.system(self._system_prompt)] + messages
        skill: Skill = await self._skill_selector(self, self._skills, messages)
        return await skill.execute(self, messages)

    async def reply(self, messages: list[Message]) -> Message:
        response = await self._llm.chat(messages)
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
