from typing import TypeVar
from pydantic import BaseModel, create_model

from .agent import Agent
from .llm import Message
from .prompts import *
from .skills import Skill
from .tools import Tool


class ChooseResponse(BaseModel):
    reasoning: str
    selection: str


class DecideResponse(BaseModel):
    reasoning: str
    answer: bool


class ToolResult(BaseModel):
    tool: str
    description: str
    result: str


class SkillSelection(BaseModel):
    reasoning: str
    skill: str


T = TypeVar("T")


class Context:
    def __init__(self, agent: Agent, messages: list[Message]):
        self.agent = agent
        self._messages = messages

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def _expand_content(self, *instructions):
        messages = self.messages

        for instruction in instructions:
            if isinstance(instruction, str):
                instruction = Message.system(instruction)

            messages.append(instruction)

        return messages

    async def reply(self, *instructions: str | Message) -> Message:
        """Reply to the provided messages.

        This method will use the LLM to generate a response to the provided messages.
        It does not use any skills.
        Mostly useful inside skills to finish the conversation.
        """
        return await self.agent.llm.chat(self._expand_content(*instructions))

    async def choose(self, options: list[T], *instructions: str | Message) -> T:
        """Choose one option out of many.

        This method will use the LLM to choose one option out of many.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        mapping = {str(option): option for option in options}

        prompt = DEFAULT_CHOOSE_PROMPT.format(
            options="\n".join([f"- {option}" for option in options]),
            format=ChooseResponse.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            ChooseResponse, self._expand_content(*instructions, Message.system(prompt))
        )

        return mapping[response.selection]

    async def decide(self, *instructions) -> bool:
        """Decide True or False.

        This method will use the LLM to decide True or False.
        It does not use any skills.
        Mostly useful inside skills to make decisions.
        """
        prompt = DEFAULT_DECIDE_PROMPT.format(
            format=DecideResponse.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            DecideResponse, self._expand_content(*instructions, Message.system(prompt))
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
            format=ChooseResponse.model_json_schema(),
        )

        response = await self.agent.llm.parse(
            ChooseResponse, self._expand_content(*instructions, Message.system(prompt))
        )

        return mapping[response.selection]

    async def engage(self, *instructions: str | Message) -> Skill:
        """
        Selects a single skill to respond to the instructions.
        This method will use the LLM to pick a skill from the list of skills.
        """
        skills: list[Skill] = self.agent._skills
        skills_map = {s.name: s for s in skills}

        prompt = DEFAULT_ENGANGE_PROMPT.format(
            skills="\n".join(
                [f"- {skill.name}: {skill.description}" for skill in skills]
            ),
            format=SkillSelection.model_json_schema(),
        )

        messages = self._expand_content(*instructions, Message.system(prompt))

        response = await self.agent.llm.parse(SkillSelection, messages)
        return skills_map[response.skill]

    async def invoke(self, tool: Tool, *instructions: str | Message) -> ToolResult:
        """
        Invokes a tool with the given instructions.
        This method will use the LLM to generate the parameters for the tool.
        The tool will then be invoked with the generated parameters.
        """
        model_cls: type[BaseModel] = create_model(tool.name, **tool.parameters())

        prompt = DEFAULT_INVOKE_PROMPT.format(
            name=tool.name,
            parameters=tool.parameters(),
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
