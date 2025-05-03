import rich.traceback
from argo import Agent, LLM, Message
import httpx
import dotenv
import os
import asyncio
import rich
import googlesearch
import markitdown


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
    """
    return await agent.reply(messages)


@agent.skill
async def question_answering(agent: Agent, messages: list[Message]) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks questions about
    factual stuff.
    """
    return await agent.reply(messages)


@agent.skill
async def summarize(agent: Agent, messages: list[Message]) -> Message:
    """Summarize content.

    Use this skill when the user asks to summarize.
    """
    return await agent.reply(messages + [
        Message.system("Summarize the content of the conversation.")
    ])


@agent.skill
async def search(agent: Agent, messages: list[Message]) -> Message:
    """Search the web for information.

    Use this skill when the user asks to search
    on the internet.

    This skill just provides unbiased information
    on a given topic.
    """
    results = await search_tool.invoke(agent, messages)
    md = markitdown.MarkItDown()
    summaries = []

    for result in results:
        client = httpx.AsyncClient()

        try:
            content = await client.get(result)
            text = md.convert_response(content)

            summary = await summarize.execute(agent, [
                Message.user(text)
            ])
            summaries.append(dict(url=result, summary=summary))
        except Exception as e:
            rich.print(e)


@agent.tool
async def search_tool(query: str) -> str:
    """Search the web for information.
    """
    return list(googlesearch.search(query, num_results=5, unique=True))


async def run():
    history = []

    while True:
        try:
            user_input = input(">>> ")
            history.append(Message.user(user_input))
            response = await agent.perform(history)
            history.append(response)
            print()
        except (EOFError, KeyboardInterrupt):
            break


asyncio.run(run())
