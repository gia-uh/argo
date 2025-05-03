from argo import Agent, LLM, Message
from argo.cli import run_sync
import dotenv
import os


dotenv.load_dotenv()


async def callback(chunk:str):
    print(chunk, end="")


agent = Agent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback),
)


@agent.skill
async def chat(agent:Agent, messages: list[Message]) -> Message:
    """Casual chat with the user.
    """
    return await agent.reply(*messages)


run_sync(agent)
