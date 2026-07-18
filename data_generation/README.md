# Response-data generation example

`generate_responses.py` is a publication-friendly version of the vLLM workflow used for the
paper. It uses the zero-shot and one-shot prompts and generation hyperparameters reported in
Appendix F: temperature 0.5, top-p 0.95, 10,240 maximum output tokens, and repetition penalty
1.05.

Input is JSONL with at least `id` and `problem` fields. Run on a CUDA machine with vLLM:

```bash
pip install -e ".[generation]"
python data_generation/generate_responses.py \
  --model Qwen/Qwen3-8B \
  --input problems.jsonl \
  --output responses.jsonl \
  --prompt zero-shot
```

The output records the raw response, extracted boxed answer, reasoning text, and reasoning token
count. Answer equivalence for olympiad mathematics is not generally safe to reduce to string
matching. Add a verified `correct` field (0 or 1) with your chosen grader, then use
`build_matrices.py` to create the accuracy and CoT-length matrices expected by LaRT.
