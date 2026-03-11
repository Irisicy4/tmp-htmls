# Self-contained environment for cocoa-agent.
# Works with both Harbor Docker (-e docker) and Harbor Modal (-e modal).
# No COPY from build context — all glue files are fetched or written inline.

FROM ghcr.io/agent-infra/sandbox:latest

# Clone cocoabench/cocoa-agent + install deps
RUN git clone --depth 1 https://github.com/cocoabench/cocoa-agent.git /cocoa-agent-src \
    && /opt/python3.12/bin/pip install -r /cocoa-agent-src/requirements.txt \
       agent-sandbox "anthropic>=0.75.0" "colorama>=0.4.6" "google-genai>=0.5.0" \
       "openai>=2.6.1" "pyyaml>=6.0.3" "websocket-client>=1.9.0" playwright --quiet \
    && /opt/python3.12/bin/playwright install chromium \
    && ln -sf /cocoa-agent-src /cocoa-agent

# --- Harbor overlay: skip_docker monkey-patch ---
COPY overlay.py /cocoa-agent-src/overlay.py

# --- agents/__init__.py overlay: multi-agent support ---
COPY agents__init__.py /cocoa-agent-src/agents/__init__.py

# --- run_task.py: multi-agent entry point ---
COPY run_task.py /cocoa-agent-src/run_task.py

# --- Config files ---
COPY harbor-config.json /cocoa-agent-src/configs/harbor-config.json
COPY skill-phase1.json /cocoa-agent-src/configs/skill-phase1.json

WORKDIR /cocoa-agent
