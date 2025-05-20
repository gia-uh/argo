from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model

from argo.tools import Tool

from .agent import ChatAgent
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


def build(agent: ChatAgent) -> FastAPI:
    """
    Builds a FastAPI app from an agent.

    This method sets up the following default routes:
     - `/` to return the agent's description.
     - `/chat` to perform the chat with the agent.

    It also sets up endpoints for each tool.

    The agent is stored in the app's state, so it can be accessed from the routes.
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
    async def chat(message: Message) -> Message:
        return await agent.perform(message)

    for tool in agent.tools:
        model_cls = build_model(tool)

        @app.post(f"/{tool.name}", response_model=model_cls)
        async def invoke_tool(params: model_cls) -> dict:
            return await tool.run(**params.dict())

    return app


def build_model(tool: Tool) -> type[BaseModel]:
    """
    Builds a Pydantic model from a tool.
    """
    return create_model(
        tool.name,
        **{k: v for k, v in tool.parameters()},
    )


def serve(agent: ChatAgent, host:str="127.0.0.1", port:int=8000):
    app = build(agent)
    import uvicorn
    uvicorn.run(app, host=host, port=port)
