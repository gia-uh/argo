import asyncio
from .agent import Agent
from .llm import Message


def run_in_cli(agent:Agent):
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
