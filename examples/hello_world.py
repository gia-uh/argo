from argo import Agent, LLM, Message
import dotenv
import os
import asyncio


dotenv.load_dotenv()


agent = Agent(
    name="Bob",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL")),
)


@agent.skill
async def converse(llm:LLM, messages: list[Message]) -> str:
    """Casual chat with the user.
    """
    return await llm.chat(messages)


async def run():
    history = []

    while True:
        try:
            user_input = input(">>> ")
            history.append(Message.user(user_input))
            response = await agent.perform(history)
            history.append(response)
            print(response.content)
        except (EOFError, KeyboardInterrupt):
            break


asyncio.run(run())
