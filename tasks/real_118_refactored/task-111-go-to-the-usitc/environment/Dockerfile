FROM ghcr.io/agent-infra/sandbox:latest
RUN /opt/python3.12/bin/pip install --upgrade \
    typing_extensions sniffio anyio \
    python-json-logger traitlets jupyter_client \
    terminado referencing jsonschema pygments \
    --target /opt/python3.12/lib/python3.12/site-packages/ \
    --quiet
