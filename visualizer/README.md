# Agent Visualization Tool

A web-based visualization tool for viewing agent execution traces, similar to UI-TARS showcase.

## Features

- **Left Panel**: Shows the complete agent trace with:
  - **Think**: Agent's reasoning/thought process
  - **Action**: Actions taken by the agent (browser, file, code, shell operations)
  - **Observation**: Feedback/results from each action

- **Right Panel**: 
  - **Content Viewer**: Displays detailed content for selected actions:
    - Browser screenshots
    - File contents
    - Code execution results
    - Shell command outputs
  - **Playback Controls**: 
    - Progress bar (draggable)
    - Play/Pause button
    - Previous/Next step navigation
    - Speed control (0.5x, 1x, 2x, 4x)
    - Reset button

## Usage

1. **(Optional) Download a run from the Modal volume** if it isn't local yet:
   ```bash
   modal volume ls  evolve-bench-results /                 # list all runs
   modal volume get evolve-bench-results /<run-name> results/ --force
   ```

2. **Start the visualization server**, pointing it at a run folder:
   ```bash
   python visualizer/server.py --data-dir results/<run-name> --port 8081
   ```
   Useful flags:
   - `--compare-dir results/<other-run>` — side-by-side comparison of two runs
   - `--host 0.0.0.0` — bind on all interfaces (default is `localhost`)

3. **Open your browser** and navigate to:
   ```
   http://localhost:8081
   ```

4. **Select a trial** from the left-panel dropdown and **navigate the trace**:
   - Click any think / action / observation block to inspect it on the right
   - Use playback controls to step through the execution
   - Drag the progress bar to jump to any step

## Data Format

The visualization tool expects result JSON files with a `visualization_data` field containing:

```json
{
  "visualization_data": {
    "task_description": "...",
    "iterations": [
      {
        "iteration": 1,
        "think": "Agent's reasoning...",
        "actions": [
          {
            "action": {
              "action_type": "browser_click",
              "x": 100,
              "y": 200
            },
            "observation": "Action executed successfully",
            "screenshot": "base64_encoded_image..."
          }
        ]
      }
    ]
  }
}
```

## Architecture

- **Backend** (`server.py`): Simple HTTP server that:
  - Serves the HTML/JS frontend
  - Provides API endpoints to load visualization data from JSON files
  - Lists available result files

- **Frontend** (`index.html`): Single-page application with:
  - Trace visualization
  - Content viewer
  - Playback controls

## Notes

- The visualization data is automatically collected during agent execution in `executor/__init__.py`
- Browser actions automatically trigger screenshots for visualization
- All screenshots are stored as base64-encoded strings in the visualization data



