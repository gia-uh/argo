DEFAULT_SYSTEM_PROMPT = """
You are {name}.

This is your description:
{description}
"""


DEFAULT_CHOOSE_PROMPT = """
Given the previous messages, you have
to select one and only one of the following items
to reply:

{options}

First provide a reasoning for your response,
and then the right selection.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_DECIDE_PROMPT = """
Given the previous messages, you have
to reply only with True or False.

First provide a reasoning for your response,
and then your answer.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_EQUIP_PROMPT = """
Given the previous messages, you have to pick
one of the following tools to invoke.

{tools}

First provide a reasoning for your response,
and then the right selection.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_INVOKE_PROMPT = """
Given the previous messages, your task
is to generate parameters to invoke the following tool.

Name: {name}.

Description:
{description}

Defaults:
{defaults}

Missing parameters:
{parameters}

Return the reasoning and parameters as a JSON object
with the following format:

{format}

Provide only the values for the parameters without defaults.
"""


DEFAULT_ENGAGE_PROMPT = """
You have the following skills:

{skills}

Given the previous messages, select the best
skill to respond to the user.

First provide a reasoning of the selection,
and then the name of the relevant skill.

Reply with a JSON object in the following format:

{format}
"""


DEFAULT_CREATE_PROMPT = """
Your task is to create an object of type {type} defined as
a Pydantic model with the following signature:

```python
{signature}
```

{docs}

Reply only with a JSON object that represents a {type} in the following format:

{format}
"""