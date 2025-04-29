from argo import Agent, LLM, Message
import dotenv
import os


dotenv.load_dotenv()


agent = Agent(
    name="Bob",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL")),
)


@agent.skill
def converse(llm:LLM, messages: list[Message]) -> str:
    return llm.chat(messages)


while True:
    user_input = input("You: ")
    response = agent.act(user_input)
    print(f"Bob: {response}")
