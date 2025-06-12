import asyncio
import sys

from pathlib import Path
import dotenv
import rich
from typer import Argument, Option, Typer, Exit
from .agent import ChatAgent
from .llm import LLM, Message
from .declarative import parse


dotenv.load_dotenv()


app = Typer(name="argo", help="Argo CLI", no_args_is_help=True)


def loop(agent: ChatAgent):
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
        while True:
            try:
                user_input = input(">>> ")
                m = Message.user(user_input)

                async for r in agent.perform(m):
                    pass

                print("\n")
            except (EOFError, KeyboardInterrupt):
                break

    asyncio.run(run())


@app.command()
def run(
    path: Path = Argument(help="A YAML file with an agent definition."),
    api_key: str = Option(
        None, "--api-key", "-k", help="API key for the LLM.", envvar="API_KEY"
    ),
    base_url: str = Option(
        None, "--base-url", "-u", help="Base URL for the LLM.", envvar="BASE_URL"
    ),
    model: str = Option(
        ..., "--model", "-m", help="Model to use for the LLM.", envvar="MODEL"
    ),
    verbose: bool = Option(False, "--verbose", "-v", help="Enable verbose mode."),
):
    """
    Run an agent defined in a YAML file with a basic CLI loop.
    """

    def callback(chunk: str):
        print(chunk, end="")

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        callback=callback,
        verbose=verbose,
    )

    config = parse(path)
    agent = config.compile(llm)
    loop(agent)


@app.command()
def serve(
    path: Path = Argument(help="A YAML file with an agent definition."),
    api_key: str = Option(
        None, "--api-key", "-k", help="API key for the LLM.", envvar="API_KEY"
    ),
    base_url: str = Option(
        None, "--base-url", "-u", help="Base URL for the LLM.", envvar="BASE_URL"
    ),
    model: str = Option(
        ..., "--model", "-m", help="Model to use for the LLM.", envvar="MODEL"
    ),
    host: str = Option("127.0.0.1", "--host", "-h", help="Host IP to bind to."),
    port: int = Option(8000, "--port", "-p", help="Port to bind to."),
    verbose: bool = Option(False, "--verbose", "-v", help="Enable verbose mode."),
):
    """
    Start a FastAPI server to run an agent in API-mode.
    """
    try:
        from .server import serve as serve_loop
    except ImportError:
        print("Please install argo[server] to use this command.")
        raise Exit(1)

    llm = LLM(model=model, api_key=api_key, base_url=base_url, verbose=verbose)

    config = parse(path)
    agent = config.compile(llm)
    serve_loop(agent, host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
