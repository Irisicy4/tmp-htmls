# Self-contained environment for cocoa-agent.
# Build context is repo root (set via docker-compose.yaml).
# Agent files live in agents/cocoa_agent/ and configs/ — no sync required.

FROM ghcr.io/agent-infra/sandbox:latest

# Clone cocoabench/cocoa-agent
RUN git clone --depth 1 https://github.com/cocoabench/cocoa-agent.git /cocoa-agent-src
RUN /opt/python3.12/bin/pip install -r /cocoa-agent-src/requirements.txt \
    agent-sandbox "anthropic>=0.75.0" "colorama>=0.4.6" "google-genai>=0.5.0" \
    "openai>=2.6.1" "pyyaml>=6.0.3" "websocket-client>=1.9.0" playwright --quiet
RUN /opt/python3.12/bin/playwright install chromium

# Copy agent glue from agents/cocoa_agent/ and configs/ (repo root)
COPY agents/cocoa_agent/run_task.py agents/cocoa_agent/overlay.py /cocoa-agent-src/
COPY agents/cocoa_agent/agents_overlay/__init__.py /cocoa-agent-src/agents/__init__.py
COPY agents/cocoa_agent/configs/ /cocoa-agent-src/configs/
COPY configs/ /cocoa-agent-src/configs/

RUN ln -sf /cocoa-agent-src /cocoa-agent
WORKDIR /cocoa-agent
