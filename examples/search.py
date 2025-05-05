from pydantic import BaseModel
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


@agent.skill
async def question_answering(ctx: Context) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """
    return await ctx.reply("Reply concisely to the user.")


class Summary(BaseModel):
    url: str
    title: str
    summary: str


async def summarize(agent: Agent, url: str, content: str) -> Message:
    return await agent.llm.parse(
        Summary,
        [
            Message.system(
                f"Summarize the following webpage.\n\nURL:{url}.\n\nProvide the summary in a JSON with format: {Summary.model_json_schema()}"
            ),
            Message.user(content),
        ]
    )


@question_answering.requires
async def search(ctx: Context) -> Message:
    results = await ctx.invoke(search_engine)
    md = markitdown.MarkItDown()
    summaries = []

    for result in results:
        if not result.startswith("http"):
            continue

        try:
            text = md.convert_url(result).markdown
        except:
            continue

        summary = await summarize(agent, result, text)
        summaries.append(str(summary))

    return Message.assistant("\n\n".join(summaries))


@agent.tool
async def search_engine(query: str) -> str:
    """Search the web for information."""
    return list(str(s) for s in googlesearch.search(query, num_results=5, unique=True))


loop(agent)
