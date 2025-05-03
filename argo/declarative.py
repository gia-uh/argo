import abc
from typing import Annotated, Any, Literal, Union
import yaml

from pydantic import BaseModel, Discriminator, Field, Tag

from argo.agent import Agent, Skill, Message
from argo.llm import LLM


class ToolConfig(BaseModel):
    name: str
    description: str


class SkillStep(BaseModel):
    @abc.abstractmethod
    def compile(self):
        pass


class DecideStep(SkillStep):
    decide: str


class ChooseStep(SkillStep):
    choose: list[str]


class ReplyStep(SkillStep):
    reply: list[str]

    def compile(self):
        async def reply_step(agent: Agent, messages: list[Message]) -> Message:
            messages = list(messages)

            for m in self.reply:
                messages.append(Message.system(m))

            return await agent.reply(*messages)

        return reply_step


def get_skill_step_discriminator_value(v: Any) -> str:
    if isinstance(v, SkillStep):
        return v.__class__.__name__
    elif isinstance(v, dict):
        if "decide" in v:
            return "DecideStep"
        elif "choose" in v:
            return "ChooseStep"
        elif "reply" in v:
            return "ReplyStep"

    raise ValueError(f"Invalid SkillStep: {v}")


class SkillConfig(BaseModel):
    name: str
    description: str
    steps: list[
        Annotated[
            Union[
                Annotated[DecideStep, Tag("DecideStep")],
                Annotated[ChooseStep, Tag("ChooseStep")],
                Annotated[ReplyStep, Tag("ReplyStep")],
            ],
            Discriminator(get_skill_step_discriminator_value),
        ]
    ]

    def compile(self) -> Skill:
        return DeclarativeSkill(self)


class DeclarativeSkill(Skill):
    def __init__(self, config: SkillConfig):
        super().__init__(config.name, config.description)
        self.steps = [s.compile() for s in config.steps]

    async def _execute(self, agent, messages):
        m: Message = None

        for step in self.steps:
            m = await step(agent, messages)
            messages.append(m)

        return m


class AgentConfig(BaseModel):
    name: str
    description: str

    tools: list[ToolConfig] = Field(default_factory=list)
    skills: list[SkillConfig]

    def compile(self, llm: LLM) -> Agent:
        agent = Agent(name=self.name, description=self.description, llm=llm)

        for skill in self.skills:
            agent.skill(skill.compile())

        return agent


def parse(path) -> AgentConfig:
    with open(path) as fp:
        return AgentConfig(**yaml.safe_load(fp))
