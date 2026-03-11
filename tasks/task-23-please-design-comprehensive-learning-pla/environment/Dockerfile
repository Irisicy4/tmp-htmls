# Self-contained environment for running agents.
# Works with both Harbor Docker (-e docker) and Harbor Modal (-e modal).

FROM ghcr.io/agent-infra/sandbox:latest

# Clone cocoabench/cocoa-agent + install deps
RUN git clone --depth 1 https://github.com/cocoabench/cocoa-agent.git /cocoa-agent-src \
    && /opt/python3.12/bin/pip install -r /cocoa-agent-src/requirements.txt \
       agent-sandbox "anthropic>=0.75.0" "colorama>=0.4.6" "google-genai>=0.5.0" \
       "openai>=2.6.1" "pyyaml>=6.0.3" "websocket-client>=1.9.0" playwright --quiet \
    && /opt/python3.12/bin/playwright install chromium \
    && ln -sf /cocoa-agent-src /cocoa-agent

# --- Harness (agent-agnostic) ---
COPY run_task.py skill_store.py skill_extractor.py /harness/
COPY adapters/ /harness/adapters/
COPY configs/ /harness/configs/
COPY skills/ /skills/

WORKDIR /harness
