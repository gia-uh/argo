from argo import Agent, LLM, Message, Context
from argo.cli import loop
import dotenv
import os


dotenv.load_dotenv()


def callback(chunk:str):
    print(chunk, end="")


agent = Agent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=True),
)


@agent.skill
async def chat(ctx: Context) -> Message:
    """Casual chat with the user.
    """
    return await ctx.reply()


loop(agent)
