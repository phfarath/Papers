"""LLM simulado determinístico + cliente OpenAI opcional.

PT: Por que um MockLLM? Não há chave de API no ambiente, então cada paper é
reimplementado de forma fiel ao MECANISMO, com um "LLM" determinístico que roteia
prompts para handlers registrados pelo marcador [[task:...]] na primeira linha.

REGRA DE HONESTIDADE: um mock handler recebe APENAS a string do prompt e um
gerador `rng`. Ele nunca pode acessar estado do ambiente, gabarito ou globais.
Assim, qualquer efeito da memória nos resultados passa pelo texto do prompt —
exatamente como com um LLM real.

EN: Why a MockLLM? There is no API key in the environment, so each paper is
reimplemented faithful to the MECHANISM, with a deterministic "LLM" that routes
prompts to handlers registered via the [[task:...]] marker on the first line.

HONESTY RULE: a mock handler receives ONLY the prompt string and an `rng`. It
must never read environment state, gold answers or globals. Hence any effect of
memory on results must flow through the prompt text — exactly like a real LLM.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Protocol

# PT: assinatura de um handler: (prompt, rng) -> texto de resposta.
# EN: handler signature: (prompt, rng) -> response text.
MockHandler = Callable[[str, random.Random], str]

_REGISTRY: dict[str, MockHandler] = {}


def task_prompt(task: str, body: str) -> str:
    """PT: prefixa o corpo com o marcador de roteamento da task.

    EN: prefixes the body with the task routing marker.
    """
    return f"[[task:{task}]]\n{body}"


def mock_handler(task: str) -> Callable[[MockHandler], MockHandler]:
    """PT: decorador que registra um handler para `task` no registry global.

    EN: decorator registering a handler for `task` in the global registry.
    """

    def wrap(fn: MockHandler) -> MockHandler:
        _REGISTRY[task] = fn
        return fn

    return wrap


def _split_task(prompt: str) -> tuple[str | None, str]:
    """PT: extrai o marcador [[task:x]] da primeira linha. EN: parse the marker."""
    first, _, _rest = prompt.partition("\n")
    if first.startswith("[[task:") and first.endswith("]]"):
        return first[len("[[task:") : -2], prompt
    return None, prompt


class LLM(Protocol):
    """PT: interface minima de LLM usada por todos os papers.

    EN: minimal LLM interface used by all papers.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...


class MockLLM:
    """PT: LLM determinístico. Roteia o prompt pelo marcador [[task:...]] para o
    handler registrado. `seed` alimenta o rng repassado aos handlers, portanto a
    mesma seed + mesmos prompts => mesmas respostas.

    EN: deterministic LLM. Routes the prompt via the [[task:...]] marker to a
    registered handler. `seed` feeds the rng passed to handlers, so the same
    seed + same prompts => same responses.
    """

    def __init__(self, seed: int = 0, handlers: dict[str, MockHandler] | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._handlers = dict(_REGISTRY)
        if handlers:
            self._handlers.update(handlers)
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        task, _ = _split_task(prompt)
        if task is None:
            raise ValueError(
                "MockLLM: prompt has no [[task:...]] marker on the first line; "
                "use task_prompt(task, body) to build prompts."
            )
        handler = self._handlers.get(task)
        if handler is None:
            raise KeyError(
                f"MockLLM: no handler registered for task {task!r}. "
                f"Available: {sorted(self._handlers)}"
            )
        self.calls.append((task, prompt))
        return handler(prompt, self._rng)


class OpenAICompatLLM:
    """PT: cliente para qualquer endpoint compatível com OpenAI.

    EN: client for any OpenAI-compatible endpoint.

    Env vars: OPENAI_API_KEY (required), OPENAI_BASE_URL (optional),
    PAPERS_LLM_MODEL (default: gpt-4o-mini).
    """

    def __init__(self, model: str | None = None) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "OpenAICompatLLM needs the 'openai' package: pip install -e '.[openai]'"
            ) from e
        self._client = openai.OpenAI()
        self.model = model or os.environ.get("PAPERS_LLM_MODEL", "gpt-4o-mini")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def get_llm(kind: str | None = None, seed: int = 0) -> LLM:
    """PT: fábrica de LLM. `kind` ou env PAPERS_LLM ∈ {mock, openai}; default mock.

    EN: LLM factory. `kind` or env PAPERS_LLM ∈ {mock, openai}; default mock.
    """
    kind = kind or os.environ.get("PAPERS_LLM", "mock")
    if kind == "mock":
        return MockLLM(seed=seed)
    if kind == "openai":
        return OpenAICompatLLM()
    raise ValueError(f"unknown llm kind {kind!r} (expected 'mock' or 'openai')")


def count_tokens(text: str) -> int:
    """PT: aproximação de contagem de tokens: max(palavras, caracteres/4).

    EN: rough token count approximation: max(words, chars/4). Documented as an
    approximation only — used for cost tables, not for billing.
    """
    return max(len(text.split()), (len(text) + 3) // 4)
