import json
from typing import TypeVar
from pydantic import BaseModel, create_model
import rich

from .agent import Agent
from .llm import Message
from .prompts import *
from .skills import Skill
from .tools import Tool
from .utils import generate_pydantic_code


class Choose(BaseModel):
    reasoning: str
    selection: str


class Decide(BaseModel):
    reasoning: str
    answer: bool


class ToolResult(BaseModel):
    tool: str
    description: str
    result: str


class Equip(BaseModel):
    reasoning: str
    tool: str


class Engage(BaseModel):
    reasoning: str
    skill: str


T = TypeVar("T")
TModel = TypeVar("TModel", bound=BaseModel)


class Context:
    def __init__(self, agent: Agent, messages: list[Message]):
        self.agent = agent
        self._messages = messages

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def _wrap(self, message: Message | str | BaseModel):
        if isinstance(message, Message):
            return message
        elif isinstance(message, str):
            return Message.system(message)
        elif isinstance(message, BaseModel):
            return Message.tool(message)

        raise TypeError(f"Invalid message type: {type(message)}")

    def _expand_content(self, *instructions):
        messages = self.messages

        for message in instructions:
            messages.append(self._wrap(message))

        return messages

    async def reply(
        self, *instructions: str | Message, persistent: bool = False
    ) -> Message:
        """Reply to the provided messages.

        This method will use the LLM to generate a response to the provided messages.
        It does not use any skills.
        Mostly useful inside skills to finish the conversation.
        """
        result = await self.agent.llm.chat(self._expand_content(*instructions))

        if persistent:
            self.add(result)

        return result

    async def choose(self, options: list[T], *instructions: str | Message) -> T:
        """Choose one option out of many.

        This method will use the LLM to choose one option out of many.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        mapping = {str(option): option for option in options}

        prompt = DEFAULT_CHOOSE_PROMPT.format(
            options="\n".join([f"- {option}" for option in options]),
            format=Choose.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            Choose, self._expand_content(*instructions, Message.system(prompt))
        )

        return mapping[response.selection]

    async def decide(self, *instructions) -> bool:
        """Decide True or False.

        This method will use the LLM to decide True or False.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        prompt = DEFAULT_DECIDE_PROMPT.format(
            format=Decide.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            Decide, self._expand_content(*instructions, Message.system(prompt))
        )

        return response.answer

    async def equip(
        self, *instructions: str | Message, tools: list[Tool] = None
    ) -> Tool:
        """Selects one and exactly one tool.

        This method will use the LLM to pick a tool from the list of tools.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        if tools is None:
            tools = self.agent._tools

        tool_str = {tool.name: tool.description for tool in tools}
        mapping = {tool.name: tool for tool in tools}

        prompt = DEFAULT_EQUIP_PROMPT.format(
            tools=tool_str,
            format=Equip.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            Equip, self._expand_content(*instructions, Message.system(prompt))
        )

        return mapping[response.selection]

    async def engage(self, *instructions: str | Message) -> Skill:
        """
        Selects a single skill to respond to the instructions.
        This method will use the LLM to pick a skill from the list of skills.
        """
        skills: list[Skill] = self.agent._skills
        skills_map = {s.name: s for s in skills}

        prompt = DEFAULT_ENGAGE_PROMPT.format(
            skills="\n".join(
                [f"- {skill.name}: {skill.description}" for skill in skills]
            ),
            format=Engage.model_json_schema(),
        )

        messages = self._expand_content(*instructions, Message.system(prompt))

        response = await self.agent.llm.parse(Engage, messages)
        return skills_map[response.skill]

    async def invoke(
        self, tool: Tool = None, *instructions: str | Message, **kwargs
    ) -> ToolResult:
        """
        Invokes a tool with the given instructions.
        This method will use the LLM to generate the parameters for the tool.
        The tool will then be invoked with the generated parameters.
        """
        if tool is None:
            tool = await self.equip(instructions)

        parameters = tool.parameters()

        for k, v in kwargs.items():
            parameters[k] = (parameters[k], v)

        tool_name_camel_case = tool.name.title().replace("_", "")
        model_cls: type[BaseModel] = create_model(tool_name_camel_case, **parameters)

        prompt = DEFAULT_INVOKE_PROMPT.format(
            name=tool.name,
            defaults=kwargs,
            parameters={k: v for k, v in parameters.items() if k not in kwargs},
            description=tool.description,
            format=model_cls.model_json_schema(),
        )

        messages = self._expand_content(*instructions, Message.system(prompt))

        response: BaseModel = await self.agent.llm.parse(
            model_cls, messages + [Message.system(prompt)]
        )

        return ToolResult(
            tool=tool.name,
            description=tool.description,
            result=str(await tool.run(**response.model_dump())),
        )

    async def create(self, *instructions: str | Message, model: type[TModel]) -> TModel:
        """
        Parses the given instructions into a model.
        This method will use the LLM to generate the parameters for the model.
        """
        messages = self._expand_content(*instructions)

        model_code = generate_pydantic_code(model)

        prompt = Message.system(DEFAULT_CREATE_PROMPT.format(
            type=model.__name__,
            signature=model_code,
            docs=model.__doc__ or "",
            format=json.dumps(model.model_json_schema(), indent=2),
        ))

        return await self.agent.llm.parse(model, messages)

    def add(self, *messages: Message | str | BaseModel) -> None:
        """
        Appends a message to the context.
        """
        for message in messages:
            self._messages.append(self._wrap(message))
