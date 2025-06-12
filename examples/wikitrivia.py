from pydantic import BaseModel
from argo import ChatAgent, LLM, Message, Context
import dotenv
import os
import wikipedia

from argo.cli import loop


dotenv.load_dotenv()


def callback(chunk: str):
    print(chunk, end="")


agent = ChatAgent(
    name="Trivial",
    description="A helpful assistant that can search Wikipedia for answering factual questions.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=True),
)


@agent.skill
async def chat(ctx: Context):
    """Casual chat with the user.

    Use this only for greetings, basic chat,
    and questions regarding your own capabilities.
    """
    yield await ctx.reply()


class Reasoning(BaseModel):
    observation: str
    thought: str
    query: str | None = None
    final: bool


class Summary(BaseModel):
    summary: str
    relevant: bool


@agent.skill
async def question_answering(ctx: Context):
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """

    ctx.add(
        """
        You have access to Wikipedia's search engine to answer the user quesstion.
        Do not assume you know anything about the question, always double check
        against Wikipedia.
        """
    )

    for i in range(5):
        reasoning = await ctx.create(
            """
            Breakdown the user request.
            First provide an observation of the
            current state of the task and the knowledge you already have.
            Then, provide a thought on the next step to take.
            Finally, provide a short, concise query for Wikipedia, if necessary.
            Otherwise, if the existing information is enough to answer, set final=True.
            """,
            model=Reasoning,
        )

        ctx.add(reasoning)

        if reasoning.final:
            await ctx.reply()
            return

        results = await ctx.invoke(search, errors="handle")

        ctx.add(results)

    await ctx.reply("Reply with the best available information in the context.")


@agent.tool
async def search(query: str, llm: LLM) -> list[Summary]:
    """Search Wikipedia for information."""
    candidates = wikipedia.search(query, results=10)

    results = []

    for title in candidates:
        page = wikipedia.page(title)
        summary = await llm.create(
            messages=[
                Message.system(
                f"""
                Given the following query, summarize the information
                in the text to answer the query.

                Query: {query}

                Reply with a JSON object with the following fields:
                - summary: A short summary of the information relevant to the query.
                - relevant: A boolean indicating if the information is relevant to the query.
                """
                ),
                Message.user(page.summary),
            ],
            model=Summary,
        )

        if summary.relevant:
            results.append(summary)

    return results


loop(agent)
