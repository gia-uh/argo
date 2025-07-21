import abc


class Skill:
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @abc.abstractmethod
    async def execute(self, ctx):
        pass


class MethodSkill(Skill):
    def __init__(self, name: str, description: str, target):
        super().__init__(name, description)
        self._target = target

    async def execute(self, ctx):
        await self._target(ctx)


async def chat(ctx):
    """
    Casual chat with the user.
    """
    await ctx.reply()
