from argo import ChatAgent, LLM
from argo.cli import loop
from argo.llm import Message
from argo.skills import chat
import dotenv
import os


dotenv.load_dotenv()


def callback(chunk:str):
    print(chunk, end="")


agent = ChatAgent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=False),
    skills=[chat],

)


loop(agent)
