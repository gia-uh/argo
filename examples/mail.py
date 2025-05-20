import asyncio
import collections
import random
from typing import Literal

from argo.agent import AgentBase
from argo.llm import LLM, Message
from pydantic import BaseModel
from argo.crew import Crew, MemoryBoard

import os
import dotenv

dotenv.load_dotenv()


class EmailStartup(BaseModel):
    username: str


class EmailItem(BaseModel):
    sender: str
    body: str


class EmailSummary(BaseModel):
    summary: str
    sentiment: str
    category: Literal['personal', 'work', 'promotional', 'subscription', 'spam']


class EmailAggregation(BaseModel):
    summary: str
    category: str


class EmailSummaries(BaseModel):
    summaries: list[EmailSummary]


llm = LLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    verbose=True,
)


class Generator(AgentBase[EmailStartup, EmailItem]):
    """
    Generates random emails.
    """

    async def process(self, input: EmailStartup):
        while True:
            category = random.choice(['personal', 'work', 'promotional', 'subscription', 'spam'])
            sender = random.choice(["Alice", "Bob", "Charlie", "Dave"])

            yield await llm.create(
                EmailItem,
                messages=[
                    Message.system(
                        f"""
Create a random {category} email example directed to {input.username} from {sender}.

Reply only in JSON format with keys sender and body.
"""
                    )
                ],
                temperature=2,
            )
            await asyncio.sleep(random.randint(1, 5))


class Summarizer(AgentBase[EmailItem, EmailSummary]):
    """
    Summarizes incoming emails.
    """

    async def process(self, input: EmailItem):
        yield await llm.create(
            EmailSummary,
            messages=[
                Message.system(
                    """
Summarize the following email.
Reply only in JSON format,
with the following keys: summary, sentiment, category.
"""
                ),
                Message.user(input.model_dump_json()),
            ],
        )


class Aggregator(AgentBase[EmailSummary, EmailAggregation]):
    """
    Aggregates email summaries into categories.
    """

    def __init__(self, min_count:int):
        self.min_count = min_count
        self.summaries = collections.defaultdict(list)

    async def process(self, input: EmailSummary):
        self.summaries[input.category].append(input)

        if len(self.summaries[input.category]) >= self.min_count:
            yield await llm.create(
                EmailAggregation,
                messages=[
                    Message.system(
                        """
Create a summary of the following emails.
Reply only in JSON format,
with the following keys: category, summary.
"""
                    ),
                    Message.user(
                        EmailSummaries(summaries=self.summaries[input.category])
                    ),
                ],
            )
            self.summaries[input.category] = []


crew = Crew(
    board=MemoryBoard(),
    agents=[Generator(), Summarizer(), Aggregator(min_count=5)],
    seed=[EmailStartup(username="John")],
)

crew.run()
