from argo import Agent, LLM, Message, Context
import dotenv
import os

from argo.cli import loop


dotenv.load_dotenv()


# Simulate the bank
class Account:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")

        self.balance -= amount
        return self.balance


account = Account(1000)


def callback(chunk: str):
    print(chunk, end="")


agent = Agent(
    name="Banker",
    description="A helpful assistant that can execute bank transactions and reply with the account information.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=True),
)


# Add a basic chat skill
@agent.skill
async def casual_chat(ctx: Context) -> Message:
    """Casual chat with the user.

    Use this skill when the user asks a general question or engages
    in casual chat.
    """
    return await ctx.reply()


@agent.skill
async def banker(ctx: Context) -> Message:
    """Interact with the bank account.

    Use this skill when the user asks for information about the bank account,
    such as balance, or asks to deposit or withdraw.
    """
    tool = await ctx.equip()
    result = await ctx.invoke(tool)
    return await ctx.reply(Message.tool(result))


@agent.tool
async def check_balance() -> dict:
    """Returns the balance in the user account.
    """
    return dict(balance=account.balance)


@agent.tool
async def deposit(ammount: int) -> dict:
    """Deposit money into the user account.
    Returns the new balance.
    """
    return dict(balance=account.deposit(ammount), deposited=ammount)


@agent.tool
async def withdraw(ammount: int) -> dict:
    """Withdraw money from the user account.
    Returns the new balance.
    """
    try:
        return dict(balance=account.withdraw(ammount), withdrawn=ammount)
    except:
        return dict(error="Insufficient funds.", balance=account.balance)


loop(agent)
