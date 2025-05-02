import abc
import functools
import inspect
from typing import TypeVar

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

Return the reasoning and parameters as a JSON object
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
            self.name, **self.parameters()
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


DEFAULT_CHOOSE_PROMPT = """
Given the previous messages, you have
to select one and only one of the following items
to reply:

{items}

First provide a reasoning for your response,
and then the right selection.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_DECIDE_PROMPT = """
Given the previous messages, you have
to reply only with True or False.

First provide a reasoning for your response,
and then your answer.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_TOOL_CHOOSE_PROMPT = """
Given the previous messages, you have to pick
one of the following tools to invoke.

{tools}

First provide a reasoning for your response,
and then the right selection.

Reply with a JSON object in the following format:

{format}
"""


T = TypeVar("T")


class ChooseResponse(BaseModel):
    reasoning: str
    selection: str


class DecideResponse(BaseModel):
    reasoning: str
    answer: bool


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
        """Main entrypoint for the agent.

        This method will select the right skill to perform the task and then execute it.
        The skill is selected based on the messages and the skills available to the agent.
        """
        messages = [Message.system(self._system_prompt)] + messages
        skill: Skill = await self._skill_selector(self, self._skills, messages)
        return await skill.execute(self, messages)

    async def reply(self, messages: list[Message]) -> Message:
        """Reply to the provided messages.

        This method will use the LLM to generate a response to the provided messages.
        It does not use any skills.
        Mostly useful inside skills to finish the conversation.
        """
        response = await self.llm.chat(messages)
        return Message.assistant(response)

    async def choose(self, options: list[T], messages: list[Message]) -> T:
        """Choose one option out of many.

        This method will use the LLM to choose one option out of many.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        options = {str(t): t for t in options}

        prompt = DEFAULT_CHOOSE_PROMPT.format(
            options="\n".join([f"- {option}" for option in options]),
            format=ChooseResponse.model_json_schema(),
        )
        response = await self.llm.parse(
            ChooseResponse, messages + [Message.system(prompt)]
        )
        return options[response.selection]

    async def decide(self, messages: list[Message]) -> bool:
        """Decide True or False.

        This method will use the LLM to decide True or False.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        prompt = DEFAULT_DECIDE_PROMPT.format(
            format=DecideResponse.model_json_schema(),
        )
        response = await self.llm.parse(
            DecideResponse, messages + [Message.system(prompt)]
        )
        return response.answer

    async def pick(self,  messages: list[Message], tools: list[Tool] = None) -> Tool:
        """Pick a tool.

        This method will use the LLM to pick a tool from the list of tools.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        if tools is None:
            tools = self._tools

        tool_str = { tool.name: tool.description for tool in tools }
        mapping = { tool.name: tool for tool in tools }

        prompt = DEFAULT_TOOL_CHOOSE_PROMPT.format(
            tools=tool_str,
            format=ChooseResponse.model_json_schema(),
        )
        response = await self.llm.parse(
            ChooseResponse, messages + [Message.system(prompt)]
        )
        return mapping[response.selection]

    def add_skill(self, skill: Skill):
        self._skills.append(skill)

    def register_tool(self, tool: Tool):
        self._tools.append(tool)

    def skill(self, target):
        if not inspect.iscoroutinefunction(target):
            raise ValueError("Skill must be a coroutine function.")

        name = target.__name__
        description = inspect.getdoc(target)
        skill = _MethodSkill(name, description, target)
        self.add_skill(skill)
        return skill

    def tool(self, target):
        # BUG: Doesn't work for sync method
        if not inspect.iscoroutinefunction(target):
            @functools.wraps(target)
            async def wrapper(*args, **kwargs):
                return target(*args, **kwargs)

            target = wrapper

        name = target.__name__
        description = inspect.getdoc(target)
        tool = _MethodTool(name, description, target)
        self.register_tool(tool)
        return tool
