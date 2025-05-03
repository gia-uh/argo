from pydantic import BaseModel
from argo import Agent, LLM, Message
import dotenv
import os
import googlesearch
import markitdown

from argo.cli import run_in_cli


dotenv.load_dotenv()


async def callback(chunk: str):
    print(chunk, end="")


agent = Agent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=True),
)


@agent.skill
async def chat(agent: Agent, messages: list[Message]) -> Message:
    """Casual chat with the user.

    Use this only for greetings and basic chat."""
    return await agent.reply(*messages)


@agent.skill
async def question_answering(agent: Agent, messages: list[Message]) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """
    return await agent.reply(*messages, Message.system("Reply concisely to the user."))


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
async def search(agent: Agent, messages: list[Message]) -> Message:
    results = await search_tool.invoke(agent, messages)
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
async def search_tool(query: str) -> str:
    """Search the web for information."""
    return list(str(s) for s in googlesearch.search(query, num_results=5, unique=True))


run_in_cli(agent)
