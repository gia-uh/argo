import asyncio
import rich
import rich.prompt
from .agent import Agent
from .llm import Message


def loop(agent:Agent):
    """Runs a CLI agent loop with integrated
    conversation history management.

    This method creates an async context internally,
    and handles storing and retrieving conversation history.
    It also handles keyboard interrupts and EOF errors.

    This method blocks the terminal waiting for user input,
    and loops until EOF (Ctrl+D) is pressed.
    """
    rich.print(f"[bold green]{agent.name}[/bold green]: {agent.description}\n")

    if agent.llm.verbose:
        rich.print(f"[yellow]Running in verbose mode.[/yellow]")

    rich.print(f"[yellow]Press Ctrl+D to exit at any time.\n[/yellow]")

    async def run():
        history = []

        while True:
            try:
                user_input = input(">>> ")
                history.append(Message.user(user_input))
                response = await agent.perform(history)
                history.append(response)
                print("\n")
            except (EOFError, KeyboardInterrupt):
                break

    asyncio.run(run())
