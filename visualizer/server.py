#!/usr/bin/env python3
"""
Simple HTTP server for serving the agent visualization interface.
"""

import json
import re
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os


def load_json_first(file_path: Path):
    """Load the first JSON object from a file (handles NDJSON / multi-object files)."""
    content = open(file_path).read()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(content)
    return obj


def load_task_meta(file_path: Path) -> dict:
    """Return minimal metadata (pass/fail, score) for a task result file."""
    try:
        data = load_json_first(file_path)
        ev = data.get("eval") or {}
        passed = ev.get("passed")
        # Extract score from feedback string "Customized Rubric Score: X.X/5"
        score = None
        feedback = ev.get("feedback", "") or ""
        m = re.search(r"(?:Rubric Score|Overall score):\s*([\d.]+)/", feedback)
        if m:
            score = float(m.group(1))
        return {"passed": passed, "score": score}
    except Exception:
        return {"passed": None, "score": None}


class VisualizationHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve visualization data."""

    def __init__(self, *args, data_dir=None, compare_dir=None, **kwargs):
        self.data_dir = Path(data_dir) if data_dir else None
        self.compare_dir = Path(compare_dir) if compare_dir else None
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/data":
            self._handle_data(parse_qs(parsed_path.query), self.data_dir)
            return

        if path == "/api/data2":
            self._handle_data(parse_qs(parsed_path.query), self.compare_dir)
            return

        if path == "/api/list":
            self._handle_list(self.data_dir)
            return

        if path == "/api/list2":
            self._handle_list(self.compare_dir)
            return

        if path == "/api/compare":
            self._handle_compare()
            return

        if path == "/" or path == "/index.html":
            self.path = "/index.html"

        return super().do_GET()

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_data(self, query_params, directory):
        file_name = query_params.get("file", [None])[0]
        if not file_name or not directory:
            self.send_error(400, "Missing file parameter or directory not configured")
            return
        file_path = directory / file_name
        if not file_path.exists():
            self.send_error(404, f"File not found: {file_name}")
            return
        try:
            data = load_json_first(file_path)
            visualization_data = data.get("visualization_data", {})
            if "eval" in data:
                visualization_data["eval"] = data["eval"]
            # Split task_description into core instruction + injected skills
            raw = visualization_data.get("task_description", "") or ""
            if "\n---\n" in raw:
                parts = raw.split("\n---\n")
                last = parts[-1].strip()
                for prefix in ("Now, here is your actual task:\n\n", "Now, here is your actual task:\n"):
                    if last.startswith(prefix):
                        last = last[len(prefix):]
                        break
                visualization_data["task_instruction"] = last
                visualization_data["task_skills"] = "\n---\n".join(parts[:-1]).strip()
            else:
                visualization_data["task_instruction"] = raw
                visualization_data["task_skills"] = ""
            self._send_json(visualization_data)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def _handle_list(self, directory):
        if not directory or not directory.exists():
            self.send_error(404, "Data directory not found")
            return
        try:
            files = []
            for f in sorted(directory.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                meta = load_task_meta(f)
                files.append({"name": f.name, **meta})
            self._send_json({"files": files, "has_compare": self.compare_dir is not None})
        except Exception as e:
            self.send_error(500, f"Error listing files: {e}")

    def _handle_compare(self):
        if not self.data_dir or not self.compare_dir:
            self._send_json({"rows": []})
            return
        try:
            rows = []
            for f in sorted(self.data_dir.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                m1 = load_task_meta(f)
                m2 = load_task_meta(self.compare_dir / f.name) if (self.compare_dir / f.name).exists() else {"passed": None, "score": None}
                rows.append({
                    "task": f.stem,
                    "p1_passed": m1["passed"],
                    "p1_score": m1["score"],
                    "p2_passed": m2["passed"],
                    "p2_score": m2["score"],
                })
            self._send_json({"rows": rows})
        except Exception as e:
            self.send_error(500, f"Error building comparison: {e}")

    def log_message(self, format, *args):
        pass


def create_handler_class(data_dir, compare_dir):
    class Handler(VisualizationHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_dir=data_dir, compare_dir=compare_dir, **kwargs)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Agent Visualization Server")
    parser.add_argument("--data-dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--compare-dir", default=None, help="Optional second directory for Phase 1 vs Phase 2 comparison")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    compare_dir = Path(args.compare_dir).resolve() if args.compare_dir else None

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return

    visualizer_dir = Path(__file__).parent
    os.chdir(visualizer_dir)

    handler_class = create_handler_class(data_dir, compare_dir)
    server = HTTPServer((args.host, args.port), handler_class)

    print(f"Visualization server: http://{args.host}:{args.port}")
    print(f"Data dir:    {data_dir}")
    if compare_dir:
        print(f"Compare dir: {compare_dir}")
    print("Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
