"""
VLLMShim — a drop-in replacement for the `llama_cpp.Llama` callable that
routes generations through a vLLM OpenAI-compatible HTTP server.

Used by:
  - eval_harness.py (offline eval — picks the shim when --vllm-url is set
    on the CLI; called from modal_vllm_eval.py for the paper's multi-seed
    canonical evaluation)
  - agent.py (live deployed Space — picks the shim when the VLLM_URL env
    var is set, which the Docker entrypoint sets after starting vllm
    serve as a subprocess on localhost:8000)

Single source of truth for the shim so the live app and the eval can't
drift in their HTTP semantics, JSON-schema constraint, stop-token handling,
or repeat-penalty mapping.

Engine-parity note vs the llama-cpp + Q4_K_M GGUF path:
  - vLLM serves Qwen2.5-3B at fp16/bf16, not Q4_K_M. Outputs are not
    bit-identical even at greedy decoding (temperature=0); the §4.2 engine
    note in solution_paper.md discloses this.
  - `grammar=<LlamaGrammar>` from llama-cpp is treated as a sentinel here:
    if any grammar is passed, we apply the rating+review JSON schema via
    vLLM's `guided_json` extra-body parameter.
  - `repeat_penalty` -> vLLM's `repetition_penalty` (extra_body)
  - `seed` is forwarded (only meaningful when temperature > 0)
"""

# The JSON schema that the rating+review prompts produce. Lives here so both
# eval_harness.py (which also feeds it into LlamaGrammar.from_json_schema for
# the llama-cpp path) and the shim use the same constraint.
RATING_REVIEW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "number", "minimum": 1, "maximum": 5},
        "review": {"type": "string", "maxLength": 300},
    },
    "required": ["rating", "review"],
}


class VLLMShim:
    """OpenAI-client-backed callable with the same signature as Llama()."""

    def __init__(self, base_url):
        from openai import OpenAI
        self.client = OpenAI(api_key="sk-no-key-needed", base_url=base_url)
        try:
            self.model_id = self.client.models.list().data[0].id
        except Exception as e:
            raise RuntimeError(
                f"VLLMShim could not list models from {base_url}: {e}"
            )

    def __call__(self, prompt, max_tokens=256, temperature=0.0, top_p=1.0,
                 repeat_penalty=1.0, seed=None, stop=None, grammar=None,
                 **kwargs):
        extra = {}
        if abs(repeat_penalty - 1.0) > 1e-6:
            extra["repetition_penalty"] = repeat_penalty
        if seed is not None:
            extra["seed"] = seed
        if grammar is not None:
            extra["guided_json"] = RATING_REVIEW_JSON_SCHEMA

        r = self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            extra_body=extra if extra else None,
        )
        return {"choices": [{"text": r.choices[0].text}]}
