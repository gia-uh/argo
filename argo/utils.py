from pydantic import BaseModel

from .agent import Agent, Skill
from .llm import Message


class SkillSelection(BaseModel):
    reasoning: str
    skill: str


DEFAULT_SKILLS_PROMPT = """
You are {agent_name}.

This is your description: {agent_description}.

You have the following skills:

{skills}

Select the right skill to perform the following task.

Reply with a JSON object in the following format:

{format}
"""


def default_skill_selector(agent:Agent, skills: list[Skill], messages: list[Message]) -> Skill:
    llm = agent._llm

    prompt = DEFAULT_SKILLS_PROMPT.format(
        agent_name=agent.name,
        agent_description=agent.description,
        skills="\n".join([f"- {skill.name}: {skill.description}" for skill in skills]),
        format=SkillSelection.model_json_schema()
    )

    skill: SkillSelection = llm.parse([SkillSelection, Message.system(prompt)] + messages)

    for s in skills:
        if s.name == skill.skill:
            return s

    raise ValueError(f"Skill {skill.skill} not found")
