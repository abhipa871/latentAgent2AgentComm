# latentAgent2AgentComm

Experimental utilities for latent agent-to-agent communication using SelfIE-style hidden-state extraction and injection.

The project extends the idea behind the SelfIE paper: instead of only asking a model to explain itself in text, pull hidden states from one generated answer and inject those states into another prompt or agent. The goal is to test whether internal representations can act like a latent message between agents, and whether another model pass can interpret or use that message without reducing everything to natural language first.

The code still exposes the Python package as `selfie_agent`, but the project framing is latent A2A communication.

## What This Explores

- Passing hidden states from one LM generation into another generation step.
- Treating intermediate activations as a communication channel between agents.
- Comparing text-level communication with latent-state communication.
- Running repeatable SelfIE-style interpretation experiments across chat models.
- Building LangGraph-compatible nodes around the hidden-state interpretation pipeline.

## Install From Git

```bash
pip install "git+https://github.com/abhipa871/latentAgent2AgentComm.git"
```

Optional LangGraph helpers:

```bash
pip install "latentAgent2AgentComm[langgraph] @ git+https://github.com/abhipa871/latentAgent2AgentComm.git"
```

## Quick Example

This example generates an answer, extracts hidden states from the answer tokens, and injects them into an interpretation pass.

```python
from selfie_agent import ModelLoader, SelfieInterpreter

model, tokenizer, _ = ModelLoader().load(
    "meta-llama/Llama-2-70b-chat-hf",
    four_bit_quant=True,
)

agent = SelfieInterpreter(model=model, tokenizer=tokenizer)
result = agent.interpret(
    original_prompt="What's the highest mountain in the world? Answer in 10 words.",
    tokens_to_interpret="all",
    source_layer=-1,
    target_layer=0,
    original_max_new_tokens=32,
    interpreter_max_new_tokens=120,
    injection_mode="aligned",
    interpretation_style="universal",
)

print(result["original_answer"])
print(result["interpretation_answers"][0])
```

## Agent-to-Agent Framing

A normal multi-agent setup usually sends text from agent A to agent B. This repo experiments with a different interface:

```text
agent A prompt
  -> agent A generation
  -> hidden states from selected tokens/layers
  -> injection into agent B interpretation/use prompt
  -> agent B output
```

That makes the repo useful for questions like:

- Can latent states preserve information that a text summary drops?
- Do different layers produce different "messages" when injected into another prompt?
- Can an interpreter agent recover useful intent, uncertainty, or reasoning traces from another agent's activations?
- How brittle is this communication channel across model families and chat templates?

## Core API

The main class is still `SelfieInterpreter`:

```python
from selfie_agent import SelfieInterpreter
```

The pipeline is:

1. Run `generate` on `original_prompt`.
2. Read hidden states from the completion.
3. Build an interpretation or receiving-agent prompt with placeholder tokens.
4. Inject selected hidden vectors at `target_layer` during generation.
5. Return the original answer, injected outputs, token/layer metadata, and raw generation outputs.

### `tokens_to_interpret`

| Form | Meaning |
| --- | --- |
| `"all"` | Use every kept generated answer token. Requires `source_layer`; use `-1` for the final hidden-state tensor. |
| `[(layer, token_idx), ...]` | Explicit hidden-state locations. Token index `0` is the first generated answer token when `answer_only=True`. |

### `injection_mode`

| Mode | Meaning |
| --- | --- |
| `"aligned"` | One generation run. Hidden vector `i` is injected at placeholder `i`. Useful when the receiving prompt has one slot per latent message token. |
| `"batch"` | One batched generation. Each row injects one selected hidden vector into every placeholder, producing one interpretation per source token. |

Useful parameters include `source_layer`, `target_layer`, `replacing_mode`, `overlay_strength`, `answer_only`, `interpretation_prompt`, `interpretation_style`, and model-specific chat-template options.

## Custom Receiving Prompts

Use `InterpretationPrompt` when you want to design the receiving agent's prompt yourself:

```python
from selfie_agent import InterpretationPrompt

interpretation_prompt = InterpretationPrompt(
    tokenizer,
    (0, 0, 0),
    placeholder="- ",
    enable_thinking=False,
    assistant_prefill="Use the latent message to summarize what the source agent was representing:",
)
```

Each `0` becomes one placeholder token where a hidden state can be injected.

## LangGraph Helpers

The package includes optional LangGraph helpers for wrapping the interpretation pipeline as a graph node:

```python
from selfie_agent import ModelLoader, SelfieInterpreter
from selfie_agent import compile_selfie_interpret_graph

model, tokenizer, _ = ModelLoader().load("meta-llama/Llama-2-7b-chat-hf", four_bit_quant=True)
agent = SelfieInterpreter(model=model, tokenizer=tokenizer)
graph = compile_selfie_interpret_graph(agent)

out = graph.invoke(
    {
        "original_prompt": "What is 2+2? One token.",
        "tokens_to_interpret": "all",
        "source_layer": -1,
        "injection_mode": "aligned",
        "interpretation_style": "universal",
    }
)
print(out["interpretation_answers"])
```

## Package Layout

- `selfie_agent.loader`: model/tokenizer loading.
- `selfie_agent.compat`: chat template styles, decoder-layer lookup, device resolution, and thinking-mode helpers.
- `selfie_agent.prompts`: `InterpretationPrompt`.
- `selfie_agent.interpreter`: hidden-state extraction, injection hooks, and `SelfieInterpreter.interpret`.
- `selfie_agent.gemma4`: optional final-channel detection for Gemma answer spans.
- `selfie_agent.generation`: generation-argument helpers and presence-penalty handling.
- `selfie_agent.langgraph_state`: LangGraph state wrappers and node compilation.

## Notes

- The repo name describes the experiment; the import path remains `selfie_agent` for compatibility.
- Hidden-state communication is experimental and model/template-sensitive.
- Use this as a research sandbox, not as a stable agent protocol.