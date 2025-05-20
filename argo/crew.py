import abc
import asyncio
import threading

from .agent import Agentic
from .llm import Message


class MessageBoard(abc.ABC):
    """
    Represents a message board.

    Provides methods to listen to messages and post messages.
    Messages are parameterized by type.
    """

    @abc.abstractmethod
    async def get[T](self, message_type: type[T]) -> T:
        pass

    @abc.abstractmethod
    async def post[T](self, message: T):
        pass


class MemoryBoard(MessageBoard):
    """
    A simple in-memory message board that uses asyncio queues.
    """

    def __init__(self):
        self.queues: dict[type, asyncio.Queue] = {}

    async def get[T](self, message_type: type[T]) -> T:
        if message_type not in self.queues:
            self.queues[message_type] = asyncio.Queue()

        return await self.queues[message_type].get()

    async def post[T](self, message: T):
        if type(message) not in self.queues:
            self.queues[type(message)] = asyncio.Queue()

        await self.queues[type(message)].put(message)


class Crew:
    """
    Represents a crew of agents.

    Each agent will be run asynchronously and will listen to messages of a specific type
    (as declared by the `Agentic` protocol).

    The crew will post messages to the message board that will be automatically
    picked up by the right agent if such agent exists.
    """

    def __init__(self, board: MessageBoard, agents: list[Agentic], seed:list=None):
        self.agents = agents
        self.board = board
        self.seed = seed or []


    async def loop(self):
        """
        Starts the crew loop.

        This method blocks in the current thread and must be called
        from an async context.

        Use `start()` if you want to run the loop in a background thread.
        """
        for item in self.seed:
            await self.board.post(item)

        try:
            await asyncio.gather(*[self._loop_agent(agent) for agent in self.agents])
        except KeyboardInterrupt:
            pass

    def start(self):
        """
        Runs `loop()` as a background async task and returns the task.
        """
        return asyncio.create_task(self.loop())

    async def _loop_agent(self, agent: Agentic):
        """
        Starts the loop for a single agent.
        """
        in_t, out_t = agent.types

        while True:
            m = await self.board.get(in_t)

            async for m in agent.perform(Message.system(m)):
                await self.board.post(m.content)

    def run(self):
        """
        Blocks the current thread until the crew loop is stopped.
        """
        asyncio.run(self.loop())
