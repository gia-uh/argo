from argo import ChatAgent, LLM, Context
from argo.cli import loop
import dotenv
import os


dotenv.load_dotenv()


def callback(chunk:str):
    print(chunk, end="")


agent = ChatAgent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=False),
)


@agent.skill
async def chat(ctx: Context):
    """Casual chat with the user.
    """
    yield await ctx.reply()


loop(agent)
