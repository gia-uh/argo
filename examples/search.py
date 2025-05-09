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


class Reasoning(BaseModel):
    observation: str
    thought: str
    action: str
    final: bool


@agent.skill
async def question_answering(ctx: Context) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """

    ctx.add("You are a helpful assistant that can answer questions about the world."
            "You have access to a search engine where you can search for information.")

    for i in range(5):
        task = await ctx.parse(
            "Breakdown the user request. First provide an observation of the"
            "current state of the task and the knowledge you already have."
            "Then, provide a thought on the next step to take. Finally,"
            "provide the action to take."
            "If the existing information is enough to answer, set final=True.",
            model=Reasoning
        )

        ctx.add(
            Message.tool(task)
        )

        if task.final:
            return await ctx.reply()

        results = await ctx.invoke(
            search,
            "Provide the simplest query that fulfills the latest action.",
        )

        ctx.add(Message.tool(results))

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
