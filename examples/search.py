from pydantic import BaseModel
import rich
from argo import Agent, LLM, Message, Context
import dotenv
import os
import googlesearch
import markitdown

from argo.cli import loop


dotenv.load_dotenv()


def callback(chunk: str):
    print(chunk, end="")


agent = Agent(
    name="Search",
    description="A helpful assistant that can search online for answering factual questions.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=True),
)


@agent.skill
async def chat(ctx: Context) -> Message:
    """Casual chat with the user.

    Use this only for greetings, basic chat,
    and questions regarding your own capabilities.
    """
    return await ctx.reply()


class TaskBreakdown(BaseModel):
    reasoning: str
    remaining: str
    next_step: str


@agent.skill
async def question_answering(ctx: Context) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """

    for i in range(5):
        task = await ctx.parse(
            "Break down the user task into smaller steps."
            "Provide the reasoning, then specify remaining part of the user task, "
            "and then the next immediate step.",
            model=TaskBreakdown
        )

        ctx.add(
            "Given the user query and current results, here is a"
            "breakdown of the task, including the remaining part of the task and the next step.",
            Message.tool(task)
        )

        results = await ctx.invoke(
            search,
            "Provide the simplest query that fullfils the next immediate step",
        )

        ctx.add(Message.tool(results))

        if await ctx.decide("Is the information sufficient to answer the user?"):
            return await ctx.reply("Reply concisely to the user.")

    return await ctx.reply(
        "Reply with the best available information in the context."
    )


@agent.tool
async def search(query: str) -> str:
    """Search the web for information."""
    candidates = googlesearch.search(query, num_results=5, unique=True)

    md = markitdown.MarkItDown()

    for result in candidates:
        if not result.startswith("http"):
            continue

        try:
            return md.convert_url(result).markdown
        except:
            continue

    return f"<ERROR: NOT FOUND '{query}'>"


loop(agent)
