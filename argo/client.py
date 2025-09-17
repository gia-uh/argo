from .agent import ChatAgent, Message
import queue
import threading
import asyncio
from typing import Iterator


def stream(agent: ChatAgent, message: str) -> Iterator[str]:
    """
    A synchronous generator that streams an ARGO agent's response,
    token by token.

    This allows integrating ARGO agents into synchronous applications
    like web servers or GUIs, where the main thread cannot be blocked
    by asynchronous code.

    Args:
        agent: The ARGO ChatAgent to use.
        user_input: The user's message to the agent.

    Yields:
        str: Tokens from the agent's response as they are generated.
    """
    token_queue = queue.Queue()

    # Temporarily override the LLM's callback to use our queue
    original_callback = agent.llm.callback
    agent.llm.callback = lambda chunk: token_queue.put(chunk)

    user_message = Message.user(message)

    async def perform_chat():
        """The async task that the background thread will run."""
        async for _ in agent.perform(user_message):
            pass
        token_queue.put(None)  # Sentinel to signal the end

    def run_in_background():
        """Thread target to run the asyncio event loop."""
        asyncio.run(perform_chat())

    # Start the agent in a background thread
    thread = threading.Thread(target=run_in_background)
    thread.start()

    # Yield tokens from the queue as they arrive
    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token

    # Wait for the thread to finish and restore the original callback
    thread.join()
    agent.llm.callback = original_callback
