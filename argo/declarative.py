import abc
from typing import Annotated, Any, Union
import yaml

from pydantic import BaseModel, Discriminator, Field, RootModel, Tag

from .agent import Agent
from .skills import Skill
from .tools import Tool
from .llm import LLM, Message
from .context import Context


class ToolConfig(BaseModel):
    name: str
    description: str


class SkillStep(BaseModel):
    @abc.abstractmethod
    def compile(self):
        pass


class DecideStep(SkillStep):
    decide: str | None
    when_true: "StepList"
    when_false: "StepList"

    def compile(self):
        true_branch = self.when_true.compile()
        false_branch = self.when_false.compile()

        async def decide_step(ctx: Context) -> Message:
            instructions = []

            if self.decide:
                instructions.append(Message.system(self.decide))

            decision = await ctx.decide(*instructions)

            if decision:
                return await true_branch(ctx)
            else:
                return await false_branch(ctx)

        return decide_step


class ChooseStep(SkillStep):
    choose: str | None = None
    choices: dict[str, "StepList"]

    def compile(self):
        compiled_choices = {k: v.compile() for k, v in self.choices.items()}

        async def choose_step(ctx: Context) -> Message:
            instructions = []

            if self.choose:
                instructions.append(Message.system(self.choose))

            choice = await ctx.choose(list(compiled_choices.keys()), *instructions)

            return await compiled_choices[choice](ctx)

        return choose_step


class ReplyStep(SkillStep):
    reply: str | None

    def compile(self):
        async def reply_step(ctx: Context) -> Message:
            instructions = []

            if self.reply:
                instructions.append(Message.system(self.reply))

            return await ctx.reply(*instructions)

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


class StepList(RootModel[
    list[
        Annotated[
            Union[
                Annotated[DecideStep, Tag("DecideStep")],
                Annotated[ChooseStep, Tag("ChooseStep")],
                Annotated[ReplyStep, Tag("ReplyStep")],
            ],
            Discriminator(get_skill_step_discriminator_value),
        ]
    ]
]):
    pass

    def compile(self):
        steps = [s.compile() for s in self.root]

        async def step_list(ctx: Context) -> Message:
            m: Message = None
            messages = ctx.messages

            for step in steps:
                m = await step(ctx)
                messages.append(m)

            return m

        return step_list


class SkillConfig(BaseModel):
    name: str
    description: str
    steps: StepList

    def compile(self) -> Skill:
        return DeclarativeSkill(self)


class DeclarativeSkill(Skill):
    def __init__(self, config: SkillConfig):
        super().__init__(config.name, config.description)
        self.steps = config.steps.compile()

    async def _execute(self, ctx):
        return await self.steps(ctx)


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
