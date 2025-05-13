import asyncio
import sys

from pathlib import Path
import dotenv
import rich
from typer import Typer, Exit
from .agent import Agent
from .llm import LLM, Message
from .declarative import parse


app = Typer(name="argo", help="Argo CLI", no_args_is_help=True)


def loop(agent: Agent):
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


@app.command()
def run(path: Path):
    """
    Run an agent in the terminal with a basic CLI loop.
    """
    dotenv.load_dotenv()
    import os

    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL = os.getenv("MODEL")

    def callback(chunk: str):
        print(chunk, end="")

    llm = LLM(model=MODEL, api_key=API_KEY, base_url=BASE_URL, callback=callback)

    config = parse(path)
    agent = config.compile(llm)
    loop(agent)


@app.command()
def serve(path: Path, host: str = "127.0.0.1", port: int = 8000):
    "Start the FastAPI server to run an agent in API-mode."
    try:
        from .server import serve as serve_loop
    except ImportError:
        print("Please install argo[server] to use this command.")
        raise Exit(1)

    dotenv.load_dotenv()
    import os

    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL = os.getenv("MODEL")

    llm = LLM(model=MODEL, api_key=API_KEY, base_url=BASE_URL)

    config = parse(path)
    agent = config.compile(llm)
    serve_loop(agent, host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
