from pydantic import BaseModel

from .agent import Agent, Skill
from .llm import Message


class SkillSelection(BaseModel):
    reasoning: str
    skill: str


DEFAULT_SKILLS_PROMPT = """
You have the following skills:

{skills}

Select the right skill to perform the following task.

Reply with a JSON object in the following format:

{format}
"""


async def default_skill_selector(agent:Agent, skills: list[Skill], messages: list[Message]) -> Skill:
    prompt = DEFAULT_SKILLS_PROMPT.format(
        skills="\n".join([f"- {skill.name}: {skill.description}" for skill in skills]),
        format=SkillSelection.model_json_schema()
    )

    skill: SkillSelection = await agent.llm.parse(SkillSelection, messages + [Message.system(prompt)])

    for s in skills:
        if s.name == skill.skill:
            return s

    raise ValueError(f"Skill {skill.skill} not found")
