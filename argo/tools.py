import inspect
import abc


class Tool:
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
    def parameters(self) -> dict[str, type]:
        pass

    @abc.abstractmethod
    async def run(self, **kwargs):
        pass


class MethodTool(Tool):
    def __init__(self, name, description, target):
        super().__init__(name, description)
        self._target = target

    def parameters(self):
        args = inspect.get_annotations(self._target)
        return {name: type for name, type in args.items() if name != "return"}

    async def run(self, **kwargs):
        return await self._target(**kwargs)
