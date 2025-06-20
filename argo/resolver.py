import functools
import inspect


class Resolver:
    """
    A simple dependency injection container.
    """
    def __init__(self) -> None:
        self.items = {}

    def register(self, item):
        self.items[type(item)] = item

    def resolve[T](self, t: type[T]) -> T:
        for tt in t.mro():
            if tt in self.items:
                return self.items[tt]

        raise ValueError(f"Could not resolve {t}")

    def wrap(self, target):
        """
        Decorator to wraps a function to automatically inject a resolver.
        Returns a new function without the Resolver parameter.
        """
        resolver_param = None
        parameters = inspect.signature(target).parameters

        for name, param in parameters.items():
            if issubclass(param.annotation, Resolver):
                resolver_param = name
                break

        if resolver_param is None:
            raise TypeError("Resolver parameter not found.")

        @functools.wraps(target)
        async def wrapper(*args, **kwargs):
            kwargs[resolver_param] = self
            return await target(*args, **kwargs)

        # Remove the LLM param from wrapper so future
        # introspection doesn't see this parameter
        wrapper.__annotations__.pop(resolver_param)

        return wrapper
