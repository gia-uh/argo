import asyncio
import rich
import rich.prompt
from .agent import Agent
from .llm import Message


def run_sync(agent:Agent):
    async def run():
        history = []

        while True:
            try:
                user_input = rich.prompt.Prompt.ask(agent.name)
                history.append(Message.user(user_input))
                response = await agent.perform(history)
                history.append(response)
                print()
            except (EOFError, KeyboardInterrupt):
                break

    asyncio.run(run())
