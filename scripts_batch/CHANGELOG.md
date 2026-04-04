# scripts_batch/ — Changelog & Key Decisions

## What this is

Custom batch runner for deprivacy-100 tasks, built on top of Harbor's multi-agent framework. Uses Modal for fire-and-forget cloud execution with UniAPI as a unified proxy for multiple LLM providers.

## Major changes from original harbor_modal_runner.py

### 1. UniAPI proxy support for CLI agents

**Problem:** Harbor's CLI agents (codex, claude-code, gemini-cli) each need their own provider API keys and authenticate directly with their respective providers.

**Solution:** Configured UniAPI as a proxy. UniAPI exposes native protocol endpoints for each provider behind a single API token:
- Codex: `OPENAI_BASE_URL=https://api.uniapi.io/v1` (OpenAI-compatible)
- Claude Code: `ANTHROPIC_BASE_URL=https://api.uniapi.io/claude` (Anthropic native)
- LLM Judge: `LLM_BASE_URL=https://api.uniapi.io/v1` (OpenAI-compatible)

**Codex specific fix:** Harbor doesn't write a `config.toml` for custom providers. We inject one into the sandbox after agent install but before run, defining `uniapi` as a custom `model_provider` with the proxy `base_url`. Without this, codex only talks to OpenAI directly.

**Status:** Codex works. Claude Code hangs silently (CLI auth issue with proxy, unresolved). Gemini CLI has no configurable base URL env var (needs real Google key).

### 2. Default model names per agent

**Problem:** Both codex (`"Model name is required"`) and claude-code failed when no model was specified. Harbor's agent factory requires `model_name` for these CLI agents.

**Solution:** Added `_DEFAULT_MODELS` dict in trigger script:
- codex: `gpt-4.1-mini`
- claude-code: `claude-sonnet-4-20250514`
- gemini-cli: `gemini-2.5-flash`

### 3. Modal app name sanitization

**Problem:** Run tags with `/` (e.g., `test-mini/codex`) were passed into Modal app names, which only allow alphanumeric, dashes, dots, underscores. Caused `"Invalid App name"` errors.

**Solution:** `run_tag.replace("/", "-")` before building the session ID.

### 4. `os.environ` override (not setdefault)

**Problem:** `os.environ.setdefault(k, v)` doesn't override existing empty values. Harbor's `create_run_agent_commands()` reads credentials from `os.environ` in the orchestrator container, not from `extra_env`.

**Solution:** Changed to `os.environ[k] = v` to force-set credentials.

### 5. Modal binary path detection

**Problem:** Collect script hardcoded `~/.local/bin/modal` but conda installs modal to the env's bin directory.

**Solution:** `shutil.which("modal")` with fallback to `~/.local/bin/modal`.

### 6. Artifact collection — `/output/` directory

**Problem:** Agents create output files in unpredictable locations. The original runner scanned `/root` and `/home` for common extensions, which picked up system files (nvm.sh, jupyter configs, etc.).

**Solution:**
- Append instruction to every task prompt telling agents to save files to `/output/` and always write `/output/result.txt`
- Collect only from `/output/` (removed `/root` `/home` fallback scan)
- Save artifacts to `results/<run-tag>/artifacts/<task-name>/` on the Modal Volume
- Add `"artifacts": [...]` field to result JSON listing collected filenames

### 7. Modal sandbox app cleanup

**Problem:** Harbor creates one Modal app per task sandbox via `App.lookup(create_if_missing=True)`. These persist as "deployed" apps and never get cleaned up. After a few runs, you hit Modal's 200 deployed apps limit.

**Solution:** Patched `_PatchedEnv.stop()` to call `app.stop.aio()` after stopping the sandbox, deleting the ephemeral app immediately.

### 8. Enriched result JSON

**Problem:** Original result only had `task`, `score`, `passed`, `feedback`. No way to group by agent, model, category, or see timing without joining multiple files.

**Solution:** Every result now includes:
- `agent`, `model`, `run_tag` — self-describing identity
- `category`, `tags` — from task.toml, enables category-level analysis
- `dimension_scores` — flattened to top level (was nested in `eval.details`)
- `evidence_summary`, `dimension_reasoning` — surfaced from judge output
- `started_at`, `finished_at` — ISO timestamps
- `artifacts` — list of collected output filenames
- `n_errors` in summary — separates infra failures from agent failures

### 9. `--force` flag for collect

**Problem:** `modal volume get` fails if destination directory exists from a previous collect.

**Solution:** Added `--force` flag to the subprocess call.

### 10. Modal API bytes/str compatibility

**Problem:** Newer Modal API returns `str` from `stdout.read()` instead of `bytes`. Calling `.decode()` on a `str` raises `"'str' object has no attribute 'decode'"`.

**Solution:** Added `_read_stdout()` helper that handles both `str` and `bytes` return types.

## What works

| Agent | UniAPI | Status |
|---|---|---|
| codex + gpt-4.1-mini | Yes | Working — scored 4.0/5 on test task |
| codex + full 100 tasks | Yes | Completed (deprivacy-codex-v2) |
| claude-code | Partial | CLI installs but hangs silently with proxy |
| gemini-cli | No | Needs real Google API key (no base URL config) |
| LLM judge | Yes | Works via UniAPI OpenAI endpoint |

## Files

| File | Purpose |
|---|---|
| `deprivacy_modal_runner.py` | Modal app — the main runner with all fixes above |
| `trigger_deprivacy.py` | Fire-and-forget trigger with UniAPI auto-detection |
| `run_all_agents.sh` | Trigger all agents in one go |
| `test_mini.sh` | 1 task x 3 agents smoke test |
| `test_mini_collect.sh` | Collect and analyze mini test results |
| `analyze_cross_agent.py` | Cross-agent comparison tables, CSV, JSON |
| `README.md` | Experiment docs and output structure |
