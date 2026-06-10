# Week 0 — Diagnostic Assessment

Purpose: establish your true baseline in Python, framework design, API
automation, and AI knowledge so the roadmap can be re-weighted. This is
**timed and closed-book where marked**. Be honest — inflated baselines only
hurt you in Week 5.

Submit by committing your answers to this folder:
- `solution_python.py` (Part A)
- `design_answers.md` (Parts B, C, D)

---

## Part A — Python Coding (60 min, no AI assistance, docs allowed)

### A1. Log Analyzer (core skills: file handling, dicts, sorting)
Write `analyze_log(path: str) -> dict` that reads a log file where each line
looks like:

```
2026-06-10 14:32:01 | ERROR | payment-service | Timeout calling /charge
2026-06-10 14:32:05 | INFO  | auth-service    | Login ok user=42
```

Return:
```python
{
    "total_lines": int,
    "by_level": {"ERROR": int, "INFO": int, ...},
    "top_service": str,          # service with most ERROR lines
    "error_messages": [str, ...] # unique ERROR messages, sorted
}
```
Requirements: handle a missing file gracefully (custom exception), skip
malformed lines but count them, use no global variables.

### A2. Retry Decorator (core skills: closures, decorators, exceptions)
Write `@retry(times=3, delay=0.1, exceptions=(ConnectionError,))` that retries
the wrapped function on the given exceptions, re-raises after the final
attempt, and preserves the wrapped function's name and docstring. Include a
short demo showing it working.

### A3. Test Data Generator (core skills: generators, JSON)
Write a generator `user_payloads(n)` that lazily yields `n` JSON-serializable
dicts: unique email, age 18–99, role cycling through ("admin", "viewer",
"editor"). Then write `save_payloads(n, path)` that streams them to a file as
JSON Lines **without** building the full list in memory.

---

## Part B — Framework Design (no time limit, write in your own words)

B1. Your current Java/Selenium/TestNG framework: describe its layers and the
**one design decision you regret**. Why?

B2. You must build API automation for a product with 12 microservices, 3
environments, and 40 engineers contributing tests. Sketch (text/tree form)
the repo structure, and explain: where does auth live, where does test data
live, how do you stop 40 people from creating 40 different HTTP clients?

B3. A test passes locally, fails in CI 30% of the time. Walk me through your
actual debugging process, step by step, and name three *root-cause categories*
with one concrete fix each. ("Add a retry" as the first answer = instant -2.)

---

## Part C — API Knowledge (15 min, closed book)

C1. What's the difference between 401 and 403? Give a test case for each.
C2. A JWT — name its three parts and two things you'd test about expiry.
C3. PUT vs PATCH — and how would your test assertions differ?
C4. You receive `200 OK` with body `{"status": "FAILED"}`. Is the API wrong?
    Argue both sides in 3 sentences.
C5. What is idempotency? Name two methods that must be idempotent and how
    you'd verify it in an automated test.

---

## Part D — AI Baseline (honesty over polish; "I don't know" is a valid answer)

D1. In your own words: what is a token, and why does an SDET testing an LLM
    feature care about context-window size?
D2. Your own app (`src/llm.py`) calls Gemini to verify helmets. The same image
    sometimes returns different answers. Is that a bug? What would you set or
    change to make automated tests against it stable, and what trade-off does
    that introduce?
D3. What is RAG (best guess)? What could go wrong between "user asks a
    question" and "app answers" — list every failure point you can think of.
D4. Rate yourself /10: prompt engineering, embeddings, AI evals tooling
    (DeepEval/Promptfoo/LangSmith), LLM safety testing.

---

## Scoring Rubric (what I'll grade)
| Area | Weight | Pass signal |
|---|---|---|
| A. Python | 40% | Pythonic, handles edge cases, no Java-isms |
| B. Framework design | 25% | Thinks in layers, ownership, scale |
| C. API | 20% | Precise, test-oriented answers |
| D. AI | 15% | Honest self-model; reasoning over recall |

Baseline ≥6/10 in an area lets us compress its weeks; <4/10 expands them.
