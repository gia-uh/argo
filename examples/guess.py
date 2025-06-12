from argo import ChatAgent, LLM, Context
from argo.cli import loop
import dotenv
import os
from pydantic import BaseModel

from argo import Context


dotenv.load_dotenv()


def callback(chunk:str):
    print(chunk, end="")


def prompt_callback():
    return input("\n?>> ")


agent = ChatAgent(
    name="Guesser",
    description="A magician that can guess what your thinking of.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=False),
    prompt_callback=prompt_callback,
)


class Reasoning(BaseModel):
    reasoning: str
    done: bool
    guess: str | None = None
    question: str | None = None


@agent.skill
async def guess(ctx: Context):
    """
    Use this skill when the user wants to play a guessing game.
    """

    for i in range(5):
        reasoning = await ctx.create(
            "Determine if you can guess what the user is thinking."
            "If so, set done=True and provide a guess."
            "Otherwise, set done=False and provide a follow up question.",
            model=Reasoning,
        )

        if reasoning.done:
            await ctx.reply(f"Claim the user is thinking of {reasoning.guess}.")
            return

        await ctx.reply(f"Ask the user {reasoning.question}.")
        await ctx.prompt()

    await ctx.reply("Give up, congratulate the user for winning.")


loop(agent)
