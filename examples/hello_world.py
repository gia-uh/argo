from argo import Agent, LLM, Message
import dotenv
import os
import asyncio


dotenv.load_dotenv()


async def callback(chunk:str):
    print(chunk, end="")


agent = Agent(
    name="Bob",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback),
)


@agent.skill
async def converse(agent:Agent, messages: list[Message]) -> str:
    """Casual chat with the user.
    """
    return await agent.reply(messages)


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
