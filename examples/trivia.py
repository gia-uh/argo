from pydantic import BaseModel
from argo import Agent, LLM, Message, Context
import dotenv
import os
import wikipedia

from argo.cli import loop


dotenv.load_dotenv()


def callback(chunk: str):
    print(chunk, end="")


agent = Agent(
    name="Trivial",
    description="A helpful assistant that can search Wikipedia for answering factual questions.",
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
    query: str | None = None
    final: bool


class SearchSummary(BaseModel):
    query: str
    summary: str


@agent.skill
async def question_answering(ctx: Context) -> Message:
    """Answer questions about the world.

    Use this skill when the user asks any questions
    that might require external knowledge.
    """

    ctx.add("You have access to Wikipedia's search engine to answer the user quesstion.")

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
            model=Reasoning
        )

        ctx.add(reasoning)

        if reasoning.final:
            return await ctx.reply()

        results = await ctx.invoke(search)

        summary = await ctx.create(
            """
            Given the desired query, sumarize the fragment of content in the
            following Wikipedia page that is related to the query.
            Make the summary as concise as possible.
            """,
            results,
            model=SearchSummary
        )

        ctx.add(summary)

    return await ctx.reply("Reply with the best available information in the context.")


@agent.tool
async def search(query: str) -> str:
    """Search Wikipedia for information."""
    candidates = wikipedia.search(query, results=1)
    return wikipedia.page(candidates[0]).content


loop(agent)
