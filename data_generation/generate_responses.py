"""Generate LaRT response records with the Appendix F vLLM configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ZERO_SHOT = """Solve the following math problem. Be clear and concise.
Problem: "{problem}"

Provide a step-by-step solution. Start each step with a number followed by a period (e.g., '1.',
'2.', etc.). Use basic LaTeX for mathematical expressions, such as for fractions, exponents, and
variables. Avoid complex formatting. At the very end of your entire response, and only at the
very end, state the final answer. This final answer must be enclosed in a single LaTeX box, like
so: \\boxed{{Your Answer}}."""

ONE_SHOT = """Solve the following math problem. Please think step-by-step to obtain the solution.
Use basic LaTeX for mathematical expressions and avoid complex formatting. At the very end of
your response, state the final answer in a single LaTeX box: \\boxed{{Your Answer}}.

Example Problem: What is the sum of the two values of x for which (x + 3)^2 = 121?
Example Solution: Expanding gives x^2 + 6x + 9 = 121, hence x^2 + 6x - 112 = 0. By Vieta's
formula, the sum of the roots is -6. Final answer: \\boxed{{-6}}.

New Problem: {problem}"""


def split_boxed_answer(text: str) -> tuple[str, str | None]:
    """Return text before the final boxed answer and its balanced-brace content."""

    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text.strip(), None
    depth = 1
    position = start + len(marker)
    while position < len(text) and depth:
        depth += (text[position] == "{") - (text[position] == "}")
        position += 1
    answer = text[start + len(marker) : position - 1].strip() if depth == 0 else None
    return text[:start].strip(), answer


def records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", choices=("zero-shot", "one-shot"), default="zero-shot")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            'Response generation requires the optional dependencies: pip install -e ".[generation]"'
        ) from exc

    items = list(records(args.input))
    template = ZERO_SHOT if args.prompt == "zero-shot" else ONE_SHOT
    prompts = [template.format(problem=item["problem"]) for item in items]
    sampling = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=10_240,
        repetition_penalty=1.05,
    )
    model = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    generations = model.generate(prompts, sampling_params=sampling)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item, generation in zip(items, generations):
            raw = generation.outputs[0].text
            reasoning, answer = split_boxed_answer(raw)
            record = {
                **item,
                "model": args.model,
                "prompt": args.prompt,
                "raw_response": raw,
                "reasoning": reasoning,
                "answer": answer,
                "cot_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False)),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
