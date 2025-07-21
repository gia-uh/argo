# ARGO

![PyPI - Version](https://img.shields.io/pypi/v/argo-ai) ![GitHub License](https://img.shields.io/github/license/apiad/argo)

**ARGO** - *Agent-based Reasoning, Governance, and Orchestration* - is a Python framework for building powerful, collaborative multi-agent systems powered by large language models (LLMs) and other AI components. Inspired by the legendary ship Argo that carried the Argonauts on their epic quest, ARGO unites diverse intelligent agents to reason, govern, and orchestrate complex workflows together.

If you want to know more about the philosophy and design principles behind ARGO, read [Towards Reliable, Consistent, and Safe LLM-based Agents](https://blog.apiad.net/p/towards-reliable-consistent-and-safe).

> NOTE: **ARGO** is a work in progress. The current state is a proof of concept and is not yet ready for production use.

## Overview

In Greek mythology, the Argo was a ship built by the master craftsman Argus and guided by the goddess Athena. It carried a crew of heroes-the Argonauts-on a daring quest for the Golden Fleece. This legendary voyage symbolizes teamwork, leadership, and the power of collective effort to overcome challenging tasks.

Similarly, **ARGO** embodies a system where multiple specialized agents collaborate under structured governance and orchestration to solve complex problems that no single agent could tackle alone.

**ARGO** is a code-first framework, meaning you create agentic workflows by writing Python code. This approach offers flexibility and control over the agentic workflows you build. However, **ARGO** also provides a very high-level, declarative interface that can be used to define agentic workflows purely with YAML files. Furthermore, **ARGO** can be run via a CLI to completely automate the execution of agentic workflows.

## Installation

**ARGO** is a very lightweight framework, with no complicated dependencies. Just install it via `pip`, `uv` or any package manager you use.

```bash
pip install argo
```

## Quick Start

**ARGO** can be used primarily in two modes: code-first, and declarative.

### Programatic mode

The programatic or code-first mode involves using the `argo` Python package in your code, and is mostly useful if you need a deep integration with your own tools.

Here is a quick hello world example that sets up a basic chat agent with no fancy tools or skills.
We assume you have the relevant environment variables `API_KEY`, `BASE_URL` and `MODEL` exported.

```python
from argo import ChatAgent, LLM
from argo.cli import loop
from argo.skills import chat
import dotenv
import os

# load environment variables
dotenv.load_dotenv()

# define callback to print chunks to console
def callback(chunk:str):
    print(chunk, end="")

# instantiate the agent
agent = ChatAgent(
    name="Agent",
    description="A helpful assistant.",
    llm=LLM(model=os.getenv("MODEL"), callback=callback, verbose=False),
    skills=[chat], # add a predefined chat skill
)

# start CLI loop
loop(agent)
```

### Declarative mode

The same behavior can be achieved with a simpler declarative interface that uses YAML files for defining skills and tools. Here is the equivalent YAML file for the above example.

```yaml
name: "Casual"
description: "An agent that performs only casual chat."
skills:
  - chat
```

You can run the above configuration with the `argo` command.

```bash
argo run <path/to/config.yaml>
```

Check the [examples](examples) folder for more detailed examples.

### Integrated API

If you install with the `server` extra (e.g., `pip install argo[server]`),
then you'll have the `argo serve` command available, that spins up a minimalistic FastAPI
server for your agent. This is useful for integrating with other services, or for building
a web-based interface for your agent.

```
argo serve <path/to/config.yaml>
```

This is not meant to be a production-ready REST server, it doesn't handle conversation
context automatically (meaning you need to mantain and pass the whole conversation in each request)
and it currently doesn't support streaming mode.

### Multi-Agent Systems

Building on top of the Agent abstraction, **ARGO** proposes a multi-agent architecture based on a typed message board. A `System` instance is a collection of agents that can communicate with each other by posting messages to a message board. The message board is typed, and agents respond to messages of the right types, and place their responses back in the same board.

This allows building complex multi-agent systems where tasks are automatically split, delegated, and coordinated among agents, with very loose coupling, as no agent needs to know the implementation details (or even the existence of) other agents.

The multi-agent system implements the `Agentic` protocol so it can be used as a single agent in another system, or fired up as a CLI or FastAPi-enabled web service.

Since **ARGO** aims to be a lightweight framework, by default it provides a development-friendly message board that is synchronous and in-memory. However, it is easy to implement a message board that is asynchronous and distributed, or that persists messages to a database, or that uses a message broker, or any other implementation that fits your needs.

In the examples, we provide a simple implementation of a message board that uses Redis a message broker and persistence backend.

## Documentation

Documentation is still under construction. However, you can check the examples for a quick start.

The following are programatic (code-based) examples:

- [Hello World](examples/hello_world.py): The barebones chat app with no extra skills.
- [Coder](examples/coder.py): A simple agent that can aswer math questions with a code interpreter.
- [Banker](examples/banker.py): A simple agent that can manage a (simulated) bank account.
- [WikiTrivia](examples/wikitrivia.py): An agent that can answer factual, multi-hop questions from Wikipedia, using the ReAct paradigm.
- [Mail](examples/email.py): A collection of agents that simulate fetching emails, summarizing and categorizing, and creating aggregated reports, all working autonomously using the Crew framework to coordinate.

The following are declarative (YAML-based) examples:

- [Hello World](examples/hello_world.yaml): The barebones chat app with no extra skills.
- [Bruno](examples/bruno.yaml): An agent that refuses to talk about Bruno.
- [Psychologist](examples/psychologist.yaml): A simplisitic agent that can counsel the user.

### Design principles

**ARGO** is designed around the key principles of modularity, flexibility, and simplicity. It is a non-opinionated framework that provides only the barebone functionality to build agentic workflows. As such, there are no concrete implementations of any specific tools or skills, or agentic paradigms like ReAct, Chain of Thought, etc.

However, **ARGO** gives you the tools to easily instantiate these paradigms and anything else you can dream of with minimal effort. Furthermore, you will find in the documentation plenty of examples using some of the most common agentic paradigms in the literature, including CoT, ReAct, actor-critic, self-consistency, ensembles, and many more.

### Key Concepts

The following is a very high-level explanation of the architecture and key ideas in **ARGO**.

#### Agents

The main concept in **ARGO** is the Agent. An agent encapsulates a set of related functionalities to solve a given domain problem. Ultimately, and agent wraps a language model (LLM) and provides a simple interface for interacting with it via customizable and dynamic prompts.

Agents can use specialized skills, including delegating work on other specialized agents. This allows you to construct hierarchies and workflows to solve complex problems.

#### Skills

Every agent has a set of one or more skills. A skill encapsulates domain knowledge on how to solve a concrete problem. This can require one or more prompts and tools, and it can be as flexible or as rigid as needed.

Think of a skill as a blueprint on how to solve a given problem, starting from a given prompt, and potentially using one or more tools. Skills can be highly structured, detailing at each each step how the agent should respond to specific inputs. However, skills can also leverage the flexibility of LLMs to be as flexible as necessary.

#### Tools

Tools encapsulate external functionality such as calling APIs, running code or commands, or performing other operations that are not directly available to the LLM. Tools are used by skills to extend the capabilities of the agent.

#### Context

A very important concept in **ARGO** is the conversation context. This object encapsulates the list of messages available in the current iteration of the conversation, and provides all the methods to interact with the language model intelligently. Furthermore, the context keeps track of where we are in the conversation flow.

#### Crew

A crew is a collection of agents that communicate via an asynchronous message board. **ARGO** provides a simple message board implemented using memory-based async queues, which is production-ready for small loads in single host setups.

Agents in a crew process messages defined via input/output types. The `Crew` class takes care of distributing each message to the correct agent, so no agent needs to be aware of the existence of other agents. This allows creating fully autonomous multi-agent systems that collaborate in a loosely coupled way.

You can seamlesly combine conversation agents, non-conversation but LLM-based agents to do batch processing or other tasks, and even non-LLM-based agents to perform tasks that do not require language processing, such as fetching data from servers or running backend tasks.

### Training Mode (under development)

A unique feature of **ARGO** is the capability to "train" agents to become better at solving specific problems. Traditionally, the cheap way to "train" LLM agents is to provide them with a set of examples of how they should behave. But crafting specific examples for each problem is time-consuming and error-prone.

Furthermore, when your agentic workflow is composed of multiple skills and tools, it becomes even more challenging to know how to craft examples that are relevant to each skill and tool, and how they interact with each other, and to maintain a coherent and relevant collection of examples to use in each possible reasoning path through the workflow.

**ARGO** aims to solve this by interactively building a set of relevant, diverse, and high-quality examples for each pathway in the agentic workflow implemented. This is done by leveraging the LLM's own capabilities to generate and refine examples, plus user guidance to evaluate how well is the agent behaving.

When in training mode, an agent will execute as usual, but it will also collect data about the execution of each skill and tool. This data is then used to generate examples that can be used to improve the agent's performance. Each training session consists of a set of interactions with the user, where the agent will ask for feedback on its behavior at different points.

When the agent deviates from the intended behavior, the user can either discard that experience, or provide natural language feedback to indicate the agent how to modify its behavior. With the right feedback, the agent can re-do the execution of the skill or tool, and try again. This process can be repeated until the agent behaves as expected, at which point the sucessfull interaction is stored.

The result of a training session is a collection of structured examples that can be later provided to a new instance of the same agent to use during inference.

## Made with ARGO

[![Made with ARGO](https://img.shields.io/badge/made_with-argo_ai-purple)](https://github.com/apiad/argo)

Submit a PR if you have a project made with **ARGO** that you want to hightlight here.

- [Lovelaice](https://github.com/apiad/lovelaice) - An AI-powered assistant for your terminal and editor.

## Roadmap

- Improve documentation and examples.
- Add tool definition via YAML and REST endpoints.
- Add streaming mode for server.
- Add support for skill composition.
- Add training mode.

## Changelog

### 0.4.0

- Skills now don't return anything, it's all handled by the context.
- Add `Context.delegate` to delegate work to another skill.

### 0.3.3

- Add method `Context.prompt`.
- Add pure completion mode for `LLM` class.
- Customizable classes for skills and tools to override default behavior.
- Customizable classes for `Context`.

### 0.3.1

- Add constraint for `Context.choose` to guarantee one option is always generated.

### 0.3.0

⭐ Agent coordination is here!

- Add `Crew` and `MessageBoard`.
- Implement simple messaging board using async queues.
- Add example of coordination (email agents).

### 0.2.4

- Add automatic FastAPI server for agents.
- Improve CLI with `typer`.
- Error handling for tool invocations.

### 0.2.3

- Improve `Context.create` to automatically inject model schema.
- Improve the search example to showcase ReAct-style reasoning.

### 0.2.2

- Redesign `choose` to define keys right in root step object.

### 0.2.1

- Add support for ReAct-style reasoning.
- Refactor the search example to showcase ReAct-style reasoning.

### 0.2.0

- Refactor the skill interface to use `Context`.
- Fix all examples.

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

Contributions are welcome! Please read the [Contribution Guidelines](CONTRIBUTING.md) for specific details.

Everyone is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md) when contributing to this project.

## License

ARGO is released under the MIT License.
