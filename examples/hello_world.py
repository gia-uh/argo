from argo import ChatAgent, LLM
from argo.cli import loop
from argo.skills import chat
import dotenv
import os


dotenv.load_dotenv()


agent = ChatAgent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), verbose=False),
    skills=[chat],

)


loop(agent)
