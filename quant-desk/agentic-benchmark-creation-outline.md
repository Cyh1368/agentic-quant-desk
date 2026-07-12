# High-Level Plan: Joe's contribution to practice-environments

Context: Joe is a Yale sophomore (Applied Math) working with Stephanie, Alfreed (afreed),
Jerry, and Matt on an early-stage AI eval lab. Goal: design agent benchmarks,
publish papers, and hopefully land a benchmark popular enough to become a startup.
Current team consensus: everyone builds evaluation tasks/datasets in their own domain.

Goal: build an agentic loop that identifies gaps in LLM benchmarking. This loop should consist of several teams, such as:
- human-in-the-loop
- advisor, which looks at a list of directions proposed by a human and delegates subagents for github PR search
- verifier, checks the proposed benchmark against what exists and determines whether what was found is unique

This is not a loop to be ran 24/7. instead, it will only run at the human user's discretion (e.g. when a new idea is added to the list of directions)

## Pick a niche no benchmark covers
Differentiate on **domain**, not just repos. Existing coverage to avoid overlapping:
SWE-bench (Python web/libs: django, scipy, sympy, flask…), SWE-bench Multimodal/
Verified, Terminal-Bench (terminal ops), Aider/polyglot (small exercises),
teammates (Python trading backend, Rust SDK, Elixir web). Candidate niches where
my applied-math background helps: physics, numerical/scientific computing correctness
(convergence, stability, precision), optimization/solver libraries, quant/
statistics libraries — bugs where the fix requires mathematical reasoning, not
just code plumbing. 

## Source candidate PRs
- Shortlist 3–5 healthy, active repos in the niche (real test suites, CI, offline-
  buildable). Prefer less-famous repos over the usual benchmark suspects.
- Mine merged PRs with `gh pr list/view/diff`: linked issue, adds/changes tests,
  moderate diff size (~30–500 lines), self-contained, no external services.
- Prefer recent PRs (post model training cutoffs) — reduces memorization risk.

## Verify novelty (no other benchmark uses it)
- Check the repo against benchmark manifests: SWE-bench/-Verified/-Multimodal
  task lists (HuggingFace datasets), Terminal-Bench task list, SWE-Gym, R2E,
  Commit0, Multi-SWE-bench (non-Python), etc. Search HuggingFace + GitHub for
  the repo name alongside "benchmark"/"eval".
- If the repo appears anywhere, either drop it or use only PRs newer than that
  benchmark's snapshot.
- Memorization spot-check: ask a frontier model to fix the issue from the issue
  text alone, no codebase; if it reproduces the exact patch, discard the PR.
- Keep a provenance note per task recording these checks (goes in the task README).

## Build 1 pilot task
- Pin repo to the PR's parent commit in a Dockerfile.
- instruction.md from the issue/PR description, rewritten so it doesn't leak the fix.
- Hidden tests adapted from the PR's tests; grade behavior, compute expected
  values independently; partial credit where the task has separable concerns.
- solution/solve.sh applying the real fix; task.toml metadata.
- Validate: base commit scores 0, oracle scores 1.0, then run a real agent
  (claude-code / codex) to confirm the task is neither trivial nor impossible
  (target: frontier agents sometimes fail).

## Review, then scale
- PR the pilot task to the repo; get feedback from teammates on
  difficulty, grading fairness, and doc style.
- Repeat Steps 3–5 to build out 3–5 tasks in the niche, reusing the pilot's
  scaffolding. Track per-task agent pass rates — the spread is what makes the
  eventual paper/benchmark interesting.