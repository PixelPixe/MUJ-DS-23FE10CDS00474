# Email Segmenter

Sorts emails into three categories using an LLM API:

| Category | What it covers |
|---|---|
| **Work** | Job, studies, career: colleagues, clients, professors, interviews, placement/university notices |
| **Personal** | Real people in your private life: family, friends, roommates |
| **Commercial** | Ads, promotions, newsletters, cold sales, receipts, OTPs, bank/subscription notices, spam, phishing |

Every email gets a category, a confidence score and a one-line reason. Low-confidence results are flagged `needs_review`.

## How it works

```
CSV / .eml files --> data_loader --> classifier --> LLM API --> validate JSON --> predictions.csv
                                        |   ^                                    + metrics.json
                                      cache |  prompts/prompts.yaml (instructions + few-shot)
                                            config.yaml (model, limits, categories)
```

1. **Load** emails from a CSV file or a folder of `.eml` files.
2. **Clean**: collapse whitespace and cut very long bodies (saves tokens).
3. **Prompt**: the prompt file gives the model category definitions, decision rules and 6 worked examples.
4. **Call the LLM** and ask for one small JSON object: `{category, confidence, reason}`.
5. **Validate** the reply. If it is not valid JSON or has an unknown category, ask again (up to `parse_retries`).
6. **Save** results, print a summary, and (if the CSV has a `label` column) print accuracy, precision/recall/F1 and a confusion matrix.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your API key inside
```

## Run

```bash
# Whole CSV (sample data has 30 labelled emails, so you also get evaluation metrics)
python main.py --csv data/sample_emails.csv

# A folder of exported .eml files
python main.py --eml-dir path/to/emails

# One email
python main.py --subject "Lunch tomorrow?" --body "Are we still on for 1 PM?" --sender "sam@gmail.com"

# Useful flags
python main.py --csv data/sample_emails.csv --limit 5 --workers 3 --no-cache -v
```

Outputs go to `outputs/`: `predictions.csv` (one row per email) and `metrics.json` (only when labels exist).

Your own CSV needs a `subject` and/or `body` column. Optional columns: `id`, `sender`, `label`.

## Configuration (`config.yaml`)

Everything adjustable lives here: provider, model, temperature, token limit, timeout, retries, categories, truncation length, confidence threshold, number of parallel workers, cache settings and file paths.

**Using a different LLM.** The default is Anthropic Claude. For OpenAI or any OpenAI-compatible service (Groq, Gemini, etc.), change three lines:

```yaml
llm:
  provider: openai
  model: <model name from that provider>
  api_key_env: OPENAI_API_KEY      # and put the key in .env
  base_url: https://api.groq.com/openai/v1    # leave null for OpenAI itself
```

## Prompt design (`prompts/prompts.yaml`)

- **Clear definitions** for each category, including the tricky edge cases (OTPs and receipts, cold sales emails, recruiters, university notices).
- **Ordered decision rules** so the model resolves ambiguity the same way every time. Purpose beats keywords.
- **Few-shot examples** sent as earlier chat turns, covering all three categories plus a transactional email and a prompt-injection attempt.
- **Strict JSON output** with a confidence guide, which is easy to parse and cheap (about 30 output tokens per email).
- **Injection defence**: email text is treated as data, wrapped in `<email>` tags, and the wrapper cannot be closed from inside the email.

## Efficiency

- Small, fast model + `temperature 0` + tiny JSON reply
- Body truncation and whitespace cleanup
- Parallel API calls (`workers`), results keep the original order
- On-disk cache keyed on model + prompt version + email text: re-runs cost zero API calls, and editing the prompt automatically invalidates old answers
- SDK-level retries for rate limits and network errors; a failed email never crashes the whole run

## Design decisions

- Automated business messages (receipts, OTPs, bank alerts, job-alert digests) are **Commercial**, because they come from a business system, not a person. Change this in the prompt file if you prefer otherwise.
- Institutional messages about your job or studies (placement notices, exam schedules) are **Work**, even when sent to many people.

## Tests

```bash
pytest
```

18 offline tests use a fake LLM (no API key needed): response parsing, retries, caching, truncation, ordering, API failures, prompt/config consistency, file loading and metric calculation.

## Project layout

```
main.py                  command-line entry point
config.yaml              settings
prompts/prompts.yaml     prompt file
segmenter/
  config.py              typed config loader
  llm_client.py          Anthropic + OpenAI-compatible clients
  prompts.py             prompt loading and message building
  classifier.py          classify, validate, retry, parallelise
  cache.py               on-disk prediction cache
  data_loader.py         CSV and .eml loading
  evaluate.py            accuracy / F1 / confusion matrix
data/sample_emails.csv   30 labelled sample emails
tests/test_segmenter.py  offline tests
```

## Limitations

Each email is judged on its own text, with no sender history. The sample set is small, so the metrics show the pipeline works but are not a benchmark. Very short emails ("ok, thanks") can be genuinely ambiguous, which is what the `needs_review` flag is for.
