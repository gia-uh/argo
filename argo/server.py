from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .llm import Message


class SkillDescription(BaseModel):
    name: str
    description: str


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: dict[str, str]


class AgentDescription(BaseModel):
    name: str
    description: str
    skills: list[SkillDescription]
    tools: list[ToolDescription]


def build(agent: Agent) -> FastAPI:
    """
    Builds a FastAPI app from an agent.

    This method sets up the following default routes:
     - `/` to return the agent's description.
     - `/chat` to perform the chat with the agent.
    """
    app = FastAPI()
    app.state.agent = agent

    @app.get("/")
    def info() -> AgentDescription:
        """
        Get the basic description of the agent.
        """
        return AgentDescription(
            name=agent.name,
            description=agent.description,
            skills=[SkillDescription(
                name=skill.name,
                description=skill.description,
            ) for skill in agent.skills],
            tools=[ToolDescription(
                name=tool.name,
                description=tool.description,
                parameters={ k:str(v) for k,v in tool.parameters() },
            ) for tool in agent.tools],
        )

    @app.post("/chat")
    async def chat(conversation: list[Message]) -> Message:
        return await agent.perform(conversation)

    return app


def serve(agent: Agent, host:str="127.0.0.1", port:int=8000):
    app = build(agent)
    import uvicorn
    uvicorn.run(app, host=host, port=port)
