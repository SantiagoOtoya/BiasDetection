"""Thin generation helper around the existing ``infer.LlamaGenerator``.

The existing ``LlamaGenerator.generate`` hard-codes the media-bias system prompt.
For the fact-checker we need to reuse the *same loaded model* with a *different*
system prompt, without modifying the existing inference file. This helper takes
an already-constructed ``LlamaGenerator`` (which owns the tokenizer/model/torch)
and runs generation with an arbitrary system prompt.
"""

from __future__ import annotations

from typing import Any


def generate(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Run one chat completion using the model loaded inside ``llm``."""
    torch = llm.torch
    tokenizer = llm.tokenizer
    model = llm.model

    max_new_tokens = max_new_tokens if max_new_tokens is not None else llm.max_new_tokens
    temperature = temperature if temperature is not None else llm.temperature
    top_p = top_p if top_p is not None else llm.top_p

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_length = input_ids.shape[-1]
        input_device = next(model.parameters()).device
        input_ids = input_ids.to(input_device)
        generate_kwargs: dict[str, Any] = {"input_ids": input_ids}
    else:
        rendered = (
            f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        input_length = inputs["input_ids"].shape[-1]
        input_device = next(model.parameters()).device
        generate_kwargs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
