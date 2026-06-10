# AI SDET Transformation Roadmap — 24 Weeks

**Mentee:** Test Analyst, 5 yrs (Java / Selenium / Appium / TestNG / API / Mobile)
**Target:** AI SDET, 20+ LPA interview-ready
**Start:** June 2026
**Mentor protocol:** No topic is "done" until you've submitted code, survived a strict review, and answered interview questions on it at 7/10 or better.

---

## Baseline Assessment (from your repo, 2026-06-10)

What the Traffic Violation Detection codebase tells me about you **today**:

**Stronger than you claimed:**
- You already use ABCs, type hints, a logging module, config management, and a
  provider-abstraction pattern (`src/llm.py` — `LLMProvider` → Gemini/OpenAI).
  This is *not* "limited Python." It's intermediate Python with gaps.
- You've already integrated vision LLMs with a voting/consensus strategy
  (`check_passengers_voting`) — that's an AI-reliability pattern most SDETs
  have never seen. We will weaponize this in interviews.

**Honest gaps (this is where the strictness starts):**
1. **Zero tests.** A 15-module production-flavored system with no `tests/`
   folder and no pytest in `requirements.txt`. For an SDET candidate this is
   the single biggest red flag in your portfolio. We fix this first.
2. **Repo hygiene:** committed `python_env/` and `Scripts/` binaries (hundreds
   of MB of .exe/.dll). No `.gitignore` discipline, no pinned dependency
   versions, no virtualenv documentation. Interviewers *will* open your GitHub.
3. **No CI.** Nothing runs automatically on push.
4. **Error handling and resilience** in the LLM layer is untested and
   unverified (timeouts, malformed JSON from the model, rate limits).

**Unique advantage we will exploit:** your own repo is a real AI system
(YOLO + OCR + LLM verification). Phases 5–8 will be built *against your own
application* — almost no candidate walks into an interview having tested an
AI system they also built.

---

## Phase → Week Mapping (weighted to your priorities)

| Priority | Focus | Weight | Weeks |
|---|---|---|---|
| P1 | Python + Framework Development | 40% | 0–9 |
| P2 | Advanced API Automation + Architecture | 30% | 6–13 (overlaps P1) |
| P3 | AI / LLM / RAG Testing | 20% | 14–21 |
| P4 | CI/CD, Docker, Cloud | 10% | 22 (+ drip-fed from Week 4) |
| — | Interview Preparation | continuous | mocks every 4 weeks; intensive 23–24 |

---

## WEEK 0 — Diagnostic (this week)
- Complete `mentorship/week-00-assessment/ASSIGNMENT.md` (Python coding,
  framework design, API design, AI knowledge quiz).
- I score each area /10 and re-weight the roadmap based on results.
- **Exit gate:** submission reviewed, baseline scores recorded.

---

## PHASE 1 — Python for SDETs (Weeks 1–5)

You know Java OOP; we translate, not re-teach. Every week produces a reusable
library component that later feeds the frameworks.

### Week 1 — Python idioms for Java developers
- Pythonic thinking: comprehensions, unpacking, `enum`, `dataclasses`,
  truthiness, EAFP vs LBYL, `pathlib`.
- Data structures the Java way vs Python way: dict/set/tuple, `collections`
  (defaultdict, Counter, deque, namedtuple).
- **Build:** `testutils/datagen.py` — test-data generator library (random
  users, payloads, boundary values).
- **Milestone:** solve 10 graded exercises without writing "Java in Python."

### Week 2 — OOP, exceptions, files, JSON
- Dunder methods, properties, classmethod/staticmethod, ABCs vs Protocols,
  composition over inheritance in framework design.
- Exception hierarchies, custom exceptions, context managers (`with`,
  `contextlib`).
- File handling + JSON: serialization, schema-shaped parsing, csv/yaml.
- **Build:** `testutils/config_loader.py` — layered config (defaults → file →
  env vars) with validation; custom exception family.
- **Milestone:** refactor one module of the traffic repo to use it.

### Week 3 — Decorators, generators, iterators
- Closures, decorators (with args, class-based), `functools.wraps`.
- Generators, lazy pipelines, `yield from` — and why pytest fixtures are
  generators.
- **Build:** `testutils/resilience.py` — `@retry(times, backoff, exceptions)`,
  `@timeit`, `@log_call` decorators. These go straight into the framework.
- **Milestone:** explain in interview language how a decorator works, line by
  line, /10 ≥ 8.

### Week 4 — Concurrency, logging, packaging
- `threading` vs `multiprocessing` vs `asyncio` — when each matters for test
  tooling; `concurrent.futures` for parallel API calls; GIL in interview depth.
- Production logging: handlers, formatters, per-test log capture, correlation
  IDs.
- Packaging: `pyproject.toml`, virtualenvs, pinned requirements.
- **Build:** parallel health-check tool that hits N endpoints concurrently
  with the Week 3 retry decorator and structured logs.
- **DevOps drip:** Git hygiene week — fix the traffic repo (.gitignore, purge
  binaries, pin requirements).

### Week 5 — Consolidation + Mock Interview #1
- Code kata battery (strings, dicts, file/JSON parsing under time pressure —
  the actual SDET screening style, not LeetCode-hard).
- **Mock interview:** 60 min Python + utilities review. Scored /10.
- **Exit gate:** ≥7/10 or Week 5 repeats with targeted drills.

---

## PHASE 2 — Pytest Framework (Weeks 6–9)  *(P1+P2 overlap begins)*

### Week 6 — Pytest core
- Test discovery, asserts + introspection, fixtures (scopes, autouse,
  factories, finalization), `conftest.py` layering.
- Parametrization: `@pytest.mark.parametrize`, indirect, dynamic params.
- **Build:** start `api-test-framework/` repo — proper skeleton.

### Week 7 — Pytest power features
- Marks (skip/xfail/custom), CLI options, `pytest.ini`/`pyproject` config,
  hooks (`pytest_collection_modifyitems`, `pytest_runtest_makereport`).
- Plugins: pytest-html, allure-pytest, pytest-rerunfailures, pytest-timeout.
- Writing your **own** plugin (a result-summary Slack/console reporter).
- **Milestone:** explain fixture resolution order and hook lifecycle cold.

### Week 8 — Parallelism + reporting + retries
- pytest-xdist: distribution modes, fixture pitfalls under parallelism,
  isolation strategies.
- Retry strategy: flaky-test policy (when retries are legitimate vs hiding
  rot) — interview gold.
- Allure reporting with steps, attachments, environment info.
- **DevOps drip:** GitHub Actions workflow running the suite on push.

### Week 9 — Project: API Automation Framework v1 + Mock Interview #2
- Complete framework against a real public API (e.g., Restful-Booker or
  reqres): layered client, services, schemas, fixtures, data factories,
  logging, Allure, retries, parallel, CI.
- **Exit gate:** I review the framework like a hostile principal engineer.
  ≥7/10 to proceed.

---

## PHASE 3 — Advanced API Testing & Architecture (Weeks 10–13)

### Week 10 — HTTP mastery + auth
- requests/httpx deep dive: sessions, adapters, timeouts done right,
  connection pooling, streaming.
- AuthN/AuthZ: API keys, Basic, OAuth2 flows (client-credentials vs auth-code),
  JWT structure/claims/expiry testing, refresh-token scenarios, negative
  security tests.

### Week 11 — Contract testing + mocking
- Schema validation (pydantic / jsonschema) as a test layer.
- Consumer-driven contracts with Pact: provider verification, broker, CI gates.
- Mocking & service virtualization: responses/respx, WireMock; when mocks lie.
- **Build:** contract suite + mock-backed negative tests in the framework.

### Week 12 — GraphQL + performance validation
- GraphQL: queries/mutations/fragments, error model, N+1 risks, testing
  strategies vs REST.
- Performance validation in functional suites: latency budgets/percentiles,
  SLO assertions; intro to locust/k6 for load smoke.
- **Build:** GraphQL test module + latency-budget assertion plugin.

### Week 13 — Test architecture + Mock Interview #3
- Architecture: layering, dependency injection in test code, environment
  abstraction, secrets handling, test-data strategy at enterprise scale,
  versioned test libraries shared across teams.
- **Project complete:** Enterprise API framework v2 (portfolio repo #1).
- **Exit gate:** system-design-style grilling: "Design API automation for a
  50-service platform." ≥7/10.

---

## PHASE 4 — AI Fundamentals (Weeks 14–15)

### Week 14 — How LLMs actually work (tester's depth)
- Transformer intuition, tokens/tokenization, context windows, temperature/
  top-p, system vs user prompts, function/tool calling, structured outputs.
- Model landscape: GPT, Gemini, Claude — APIs, capabilities, pricing/latency
  trade-offs (you already call Gemini/OpenAI in `src/llm.py`; now you'll
  understand every parameter you've been passing).
- **Build:** small harness that calls 2 providers with controlled temperature
  and compares output stability across N runs.

### Week 15 — Embeddings, vector DBs, RAG
- Embeddings and semantic similarity (cosine sim — you'll compute it by hand
  once), vector stores (Chroma/FAISS), chunking, retrieval, the full RAG loop.
- Prompt engineering as a *testable artifact*: prompt versioning, prompt
  regression.
- **Build:** tiny RAG app over your own repo's docs — this becomes the system
  under test for Week 19.

---

## PHASE 5 — AI Testing (Weeks 16–17)

### Week 16 — Functional AI testing + hallucination
- Why AI testing is different: non-determinism, no single oracle, eval-driven
  testing vs assertion-driven testing.
- Oracle strategies: golden answers, semantic similarity thresholds,
  LLM-as-judge (and its failure modes), property-based checks, voting —
  you already built voting in `src/llm.py`; now you'll formalize it.
- Hallucination detection: faithfulness vs factuality, claim extraction.
- **Build:** test suite for your traffic app's LLM verifications
  (`verify_helmet`, `read_plate`) with a labeled mini-dataset and accuracy
  thresholds.

### Week 17 — Safety & adversarial testing
- Prompt injection (direct/indirect), jailbreak taxonomies, toxicity testing,
  bias testing (counterfactual pairs), PII leakage, OWASP LLM Top 10.
- Responsible scoping: red-teaming your *own* authorized systems.
- **Build:** adversarial test pack — injection corpus + automated detection of
  unsafe responses, run against your Week 15 RAG app.

---

## PHASE 6 — AI Evaluation Tools (Week 18)
- **DeepEval:** metrics (answer relevancy, faithfulness, hallucination,
  G-Eval custom metrics), pytest-native integration, datasets.
- **Promptfoo:** config-driven evals, side-by-side model comparison, CI mode,
  red-team mode.
- **LangSmith:** tracing, datasets, online evals, monitoring vs testing.
- **Build:** evaluation pipeline measuring correctness, relevance,
  faithfulness, latency, and cost per run; results trend in CI.

---

## PHASE 7 — RAG Testing (Week 19)
- Component-level: retrieval metrics (precision/recall@k, MRR), chunking
  validation, embedding-drift checks, citation/grounding validation,
  context-relevance scoring.
- End-to-end: the RAG triad (context relevance, groundedness, answer
  relevance) with DeepEval/RAGAS-style metrics.
- **Project:** full test suite for your Week 15 RAG app — retrieval tests +
  generation tests + regression dataset.

---

## PHASE 8 — Capstone: AI Testing Framework (Weeks 20–21)
Portfolio repo #2 — the interview centerpiece:
- Prompt regression testing, response validation (schema + semantic),
  hallucination detection, DeepEval integration, multi-model comparison,
  cost/latency tracking, HTML/Allure reporting, GitHub Actions pipeline with
  quality gates.
- Target systems: your traffic app's LLM layer **and** the RAG app.
- Deliverables: README that sells it, architecture diagram, demo script,
  a 5-minute walkthrough you can perform in interviews.
- **Exit gate:** capstone defense — I attack the design for 45 minutes.

---

## PHASE 9 — DevOps for SDETs (Week 22)
(Most of this was drip-fed; this week consolidates.)
- Git: rebase vs merge, bisect, hooks, PR discipline.
- GitHub Actions: matrices, caching, artifacts, secrets, scheduled eval runs.
- Jenkins: pipeline-as-code essentials (declarative Jenkinsfile) — enough to
  answer interview questions credibly.
- Docker: images vs containers, Dockerfile for the test framework,
  docker-compose for SUT+tests, running suites in containers in CI.
- AWS basics: S3 (artifacts/reports), EC2, IAM concepts, Bedrock awareness.

---

## PHASE 10 — Interview Preparation (Weeks 23–24)

### Week 23 — Technical gauntlet
- Daily mocks: Python coding (live), pytest/framework deep-dive, API testing
  scenarios, AI testing scenarios ("How would you test a chatbot for a
  bank?"), each scored /10 with written feedback.

### Week 24 — Senior-signal week
- AI SDET system design: "Design the QA strategy for an LLM-powered product"
  end to end (functional + evals + safety + monitoring + CI).
- Resume rewrite around the two portfolio repos + this repo's transformation.
- Storytelling: 8 STAR stories mapped to your real work.
- Salary negotiation prep for the 20+ LPA band.
- **Final exit gate:** full-loop mock (coding + framework + AI testing +
  system design + behavioral) ≥ 8/10 average.

---

## Standing Rules
1. Every week ends with a commit to this repo (or a portfolio repo) — code,
   not notes. No commit = week not done.
2. Mock interviews every ~4 weeks (Weeks 5, 9, 13, 18, 21, 23–24). Scores are
   recorded below; the roadmap re-weights after each.
3. Reviews are strict: I flag naming, structure, error handling, and missing
   tests, not just correctness.
4. Java→Python translation is allowed for thinking, banned in code style.

## Scoreboard
| Checkpoint | Python | Framework | API | AI Testing | Notes |
|---|---|---|---|---|---|
| Week 0 baseline | – | – | – | – | pending diagnostic |
