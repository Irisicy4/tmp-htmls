#!/bin/bash
# Oracle: produces /logs/agent/result.json for verifier (Harbor REQUIRED).

set -e
mkdir -p /logs/agent

/opt/python3.12/bin/python - <<'PYEOF'
import json

task_result = """## Backpack Research Under $75 (Similar to Amazon B09YRC9Y3G)

**Reference product** (https://www.amazon.com/dp/B09YRC9Y3G): Travel laptop backpack with 15.6" laptop compartment, USB charging port, multiple organizational pockets, water-resistant material.

### Alternatives Under $75:

1. **Mancro Anti-Theft** – $49.99: 15.6" laptop compartment, USB port, water-resistant.
2. **Targus Work EcoSmart** – $59.99: 15.6" sleeve, USB passthrough, recycled materials.
3. **KROSER Travel Laptop Backpack** – $39.99: Fits 15.6", USB port, water-resistant.
4. **Bagsmart Business** – $45.99: 15.6" compartment, USB port, anti-theft design.

All under $75 with laptop compartment, USB port, water resistance."""

result = {
    "status": "success",
    "task_name": "task-01-im-looking-for-backpack-under",
    "task_result": task_result,
    "conversation": [{"role": "assistant", "content": task_result}],
}
with open("/logs/agent/result.json", "w") as f:
    json.dump(result, f, indent=2)
PYEOF
