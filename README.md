# ARGO

![PyPI - Version](https://img.shields.io/pypi/v/argo-ai) ![GitHub License](https://img.shields.io/github/license/apiad/argo)



**ARGO** - *Agent-based Reasoning, Governance, and Orchestration* - is a Python framework for building powerful, collaborative multi-agent systems powered by large language models (LLMs) and other AI components. Inspired by the legendary ship Argo that carried the Argonauts on their epic quest, ARGO unites diverse intelligent agents to reason, govern, and orchestrate complex workflows together.

## Overview

In Greek mythology, the Argo was a ship built by the master craftsman Argus and guided by the goddess Athena. It carried a crew of heroes-the Argonauts-on a daring quest for the Golden Fleece. This legendary voyage symbolizes teamwork, leadership, and the power of collective effort to overcome challenging tasks.

Similarly, **ARGO** embodies a system where multiple specialized agents collaborate under structured governance and orchestration to solve complex problems that no single agent could tackle alone.

**ARGO** is a code-first framework, meaning you create agentic workflows by writing Python code. This approach offers flexibility and control over the agentic workflows you build. However, **ARGO** also provides a very high-level, declarative interface that can be used to define agentic workflows purely with YAML files. Furthermore, **ARGO** can be run via a CLI to completely automate the execution of agentic workflows.

## Key Concepts

### Agent-based Reasoning
Each agent in ARGO is an autonomous entity capable of independent reasoning, perception, and action. Agents leverage large language models and AI tools to interpret data, make decisions, and contribute specialized expertise to the collective.

### Governance
ARGO incorporates a governance layer that structures agent interactions, defines roles and responsibilities, and enforces protocols to ensure alignment, compliance, and accountability within the multi-agent system.

### Orchestration
The orchestration component manages communication, task allocation, and workflow execution among agents. It supports flexible collaboration patterns, from linear pipelines to dynamic, adaptive workflows.

## Features

> NOTE: ARGO is a work in progress. The current state is a proof of concept and is not yet ready for production use.

- **Multi-agent collaboration:** Build teams of LLM-powered agents working in concert.
- **Structured governance:** Define organizational models and enforce collaboration protocols.
- **Flexible orchestration:** Coordinate complex workflows with customizable communication and task delegation.
- **Extensible architecture:** Easily add new agent types, tools, and interaction patterns.
- **Pythonic API:** Intuitive interfaces designed for rapid prototyping and deployment.

## Installation

**ARGO** is a very lightweight framework, with no complicated dependencies. Just install it via `pip`, `uv` or any package manager you use.

```bash
pip install argo
```

## Quick Start

**ARGO** can be used primarily in two modes: code-first, and declarative.

### Code-first mode

The code-first mode involves using the `argo` Python package in your code, and is mostly useful if you need a deep integration with your own tools.

Here is a quick hello world example that sets up a basic chat agent with no fancy tools or skills.
We assume you have the relevant environment variables `API_KEY`, `BASE_URL` and `MODEL` exported.

```python
from argo import Agent, LLM, Message
from argo.cli import loop
import dotenv
import os

# load environment variables
dotenv.load_dotenv()

# set a basic callback to print LLM respondes to terminal
def callback(chunk:str):
    print(chunk, end="")

# initialize the agent
agent = Agent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback),
)

# basic skill that just replies to user messages
# notice skills *must* be async methods for now
@agent.skill
async def chat(agent:Agent, messages: list[Message]) -> Message:
    """Casual chat with the user.
    """
    return await agent.reply(*messages)

# this sets up the chat history and conversation loop
loop(agent)
```

### Declarative mode

The same behavior can be achieved with a simpler declarative interface that uses YAML files for defining skills and tools. Here is the equivalent YAML file for the above example.

```yaml
name: "Casual"
description: "An agent that performs only casual chat."
skills:
  - name: "chat"
    description: "Casual chat with the user."
    steps:
      - reply: []
```

You can run the above configuration with the `argo` command.

```bash
argo <path/to/config.yaml>
```

## Documentation

Documentation is still under construction. However, you can check the examples for a quick start.

The following are code-first examples:

- [Hello World](examples/hello_world.py): The barebones chat app with no extra skills.
- [Coder](examples/coder.py): A simple agent that can aswer math questions with a code interpreter.
- [Banker](examples/banker.py): A simple agent that can manage a (simulated) bank account.
- [Search](examples/search.py): An agent that can answer factual questions by searching the web.

The following are YAML-first examples:

- [Hello World](examples/hello_world.yaml): The barebones chat app with no extra skills.
- [Bruno](examples/bruno.yaml): An agent that refuses to talk about Bruno.
- [Psychologist](examples/psychologist.yaml): A simplisitic agent that can counsel the user.

## Changelog

### Roadmap

- Improve documentation and examples.
- Add skill definition via YAML.
- Add tool definition via YAML and REST endpoints.
- Add support for skill composition.
- Add support for multi-agent collaboration and delegation.

### 0.1.8

- Support for choice prompts in YAML mode.
- Example for `choice` instructions.

### 0.1.7

- Support for decision prompts in YAML mode.
- Example for `decide` instruction.

### 0.1.6

- Basic API for declarative agents (YAML mode).
- Example of basic YAML agent.
- CLI entrypoint for loading YAML agents.

### 0.1.5

- Middleware for skills (`@skill.require` syntax)
- Better handling of errors.

### 0.1.4

- Verbose mode for LLM.
- Several new examples.

### 0.1.3

- Automatic skill selection and tool invocation.

### 0.1.2

- Basic architecture for agents, skills, and tools.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

ARGO is released under the MIT License.
