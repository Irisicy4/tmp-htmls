#!/usr/bin/env python3
"""Build a single self-contained HTML visualizer for a sweep run.

One screen == one task. Arrow keys navigate. Panels: Full Prompt, Scores
(broken down), Artifact, Agent trace (the compact digest the effectiveness
judge actually sees), and Judge trace.

Usage:
  python build_visualizer.py <run_dir> [<tasks_dir>] [-o out.html]
"""
from __future__ import annotations
import json, os, re, sys, argparse, html


def natkey(name: str):
    # task-9 < task-88 < task-112 ; ar-/word tasks sort after numbered
    m = re.match(r"task-(\d+)", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, 0, name)


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


_FILE_RE = re.compile(r"=== FILE:\s*([^=\r\n]+?)\s*===\s*\r?\n([\s\S]*?)\r?\n=== END FILE ===", re.I)


def extract_file_artifacts(task_result: str):
    out = []
    for m in _FILE_RE.finditer(task_result or ""):
        out.append({"name": m.group(1).strip(), "body": m.group(2)})
    return out


def build_compact_trace(task_result: str, agent_stdout: str):
    """Reproduce exactly what effectiveness.py feeds the judge:
    agent_response[:8000] head + agent_stdout[-12000:] tail."""
    head = (task_result or "")[:8000]
    s = (agent_stdout or "").strip()
    tail = s[-12000:] if len(s) > 12000 else s
    return head, tail, len(s)


def dimensions_from_eff(eff: dict):
    """Effectiveness judge stores each dimension as a top-level numeric key,
    with matching text in dimension_reasoning. Recover (name, score, reason)."""
    if not isinstance(eff, dict):
        return []
    reasoning = eff.get("dimension_reasoning") or {}
    names = list(reasoning.keys())
    if not names:  # fall back to any numeric top-level keys that aren't meta
        meta = {"overall_score", "passed"}
        names = [k for k, v in eff.items()
                 if isinstance(v, (int, float)) and k not in meta]
    dims = []
    for n in names:
        dims.append({"name": n, "score": eff.get(n), "reason": reasoning.get(n, "")})
    return dims


def build_record(run_dir, tasks_dir, task):
    d = os.path.join(run_dir, "by_task", task)
    verdict = read_json(os.path.join(d, "verdict.json")) or {}
    jt = read_json(os.path.join(d, "judge_trace.json")) or {}
    modal = read_json(os.path.join(d, "modal_verdict.json")) or {}
    # If this task was re-graded after the sweep (extraction fix / rerun),
    # the corrected judge output lives in regrade_verdict.json — prefer it so
    # the judge-trace panel matches the score shown in verdict.json.
    regrade = read_json(os.path.join(d, "regrade_verdict.json"))
    if isinstance(regrade, dict):
        jt = regrade.get("details", regrade) or jt
    feedback = read(os.path.join(d, "feedback.txt"))
    agent_stdout = read(os.path.join(d, "agent_stdout.txt"))

    task_result = modal.get("task_result") or ""
    head, tail, full_len = build_compact_trace(task_result, agent_stdout)

    # prompt: prefer the full two-turn review prompt
    prompt = ""
    if tasks_dir:
        for fn in ("review_full_prompt.txt", "modified_review_full_prompt.txt", "instruction.md"):
            p = os.path.join(tasks_dir, task, fn)
            if os.path.exists(p):
                prompt = read(p)
                break

    eff = jt.get("effectiveness_judge") or {}
    eff_prior = jt.get("effectiveness_judge_prior") or {}
    fr = jt.get("faithfulness_report") or {}

    return {
        "task": task,
        "final_score": verdict.get("final_score"),
        "passed": bool(verdict.get("passed")),
        "json_parsed": verdict.get("json_parsed"),
        "two_turn_fired": verdict.get("two_turn_fired"),
        "prompt_mode": verdict.get("prompt_mode"),
        "faith_matched": verdict.get("faith_matched"),
        "faith_total": verdict.get("faith_total"),
        "two_phase_score": verdict.get("rev84_two_phase_score"),
        "regrade_note": verdict.get("regrade_note"),
        "components": jt.get("score_components") or {},
        "llm_overall": jt.get("llm_overall_score"),
        "hard_pass_rate": jt.get("hard_constraint_pass_rate"),
        "hard_report": jt.get("hard_constraint_report") or [],
        "summary_json_source": jt.get("summary_json_source"),
        "summary_json": jt.get("summary_json"),
        "evidence_summary": eff.get("evidence_summary") or "",
        "dimensions": dimensions_from_eff(eff),
        "dimensions_prior": dimensions_from_eff(eff_prior),
        "faith": {k: v for k, v in fr.items() if k != "details"},
        "faith_details": fr.get("details") or [],
        "prompt": prompt,
        "artifacts": extract_file_artifacts(task_result),
        "trace_head": head,
        "trace_tail": tail,
        "trace_full_len": full_len,
        "feedback": feedback,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("tasks_dir", nargs="?", default="")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    bt = os.path.join(args.run_dir, "by_task")
    tasks = sorted([t for t in os.listdir(bt) if os.path.isdir(os.path.join(bt, t))], key=natkey)
    records = [build_record(args.run_dir, args.tasks_dir, t) for t in tasks]

    summary = read_json(os.path.join(args.run_dir, "_summary.json")) or {}
    out = args.out or os.path.join(args.run_dir, "visualizer.html")

    payload = json.dumps({"records": records, "summary": summary}, ensure_ascii=False)
    # neutralise any </script> inside data
    payload = payload.replace("</", "<\\/")

    htmldoc = HTML_TEMPLATE.replace("__DATA__", payload).replace(
        "__RUNNAME__", html.escape(os.path.basename(args.run_dir.rstrip("/"))))
    with open(out, "w", encoding="utf-8") as f:
        f.write(htmldoc)
    print(f"wrote {out}  ({len(records)} tasks, {len(htmldoc)//1024} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sweep Visualizer — __RUNNAME__</title>
<style>
  :root{
    --bg:#0f1117; --panel:#181b24; --panel2:#1f2330; --border:#2a2f3d;
    --fg:#e6e9ef; --dim:#98a0b3; --accent:#6ea8fe; --good:#3fb950; --bad:#f85149;
    --warn:#d29922; --mono:'SF Mono',ui-monospace,'JetBrains Mono',Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px}
  #topbar{display:flex;align-items:center;gap:14px;padding:8px 16px;background:var(--panel2);
    border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10;height:52px}
  #topbar .nav{cursor:pointer;user-select:none;font-size:22px;color:var(--accent);padding:0 6px;line-height:1}
  #topbar .nav:hover{color:#fff}
  #counter{font-variant-numeric:tabular-nums;color:var(--dim);min-width:64px}
  #taskname{font-weight:600;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:0 1 auto;max-width:38vw}
  select{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:5px 8px;font-size:12px;max-width:280px}
  .badge{padding:2px 9px;border-radius:20px;font-weight:700;font-size:12px}
  .badge.pass{background:rgba(63,185,80,.15);color:var(--good);border:1px solid var(--good)}
  .badge.fail{background:rgba(248,81,73,.13);color:var(--bad);border:1px solid var(--bad)}
  .chip{padding:2px 8px;border-radius:6px;font-size:11px;background:var(--panel);border:1px solid var(--border);color:var(--dim)}
  .chip.on{color:var(--good);border-color:var(--good)}
  .chip.off{color:var(--bad);border-color:var(--bad)}
  .grow{flex:1}
  .hint{color:var(--dim);font-size:11px}
  #board{display:grid;
    grid-template-columns:1.55fr 1.05fr 1fr;
    grid-template-rows:1fr 1fr;
    grid-template-areas:
      "prompt trace   scores"
      "prompt artifact judge";
    gap:10px;padding:10px;height:calc(100vh - 52px)}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;display:flex;flex-direction:column;overflow:hidden;min-height:0;min-width:0}
  .panel h2{margin:0;padding:8px 12px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
    color:var(--dim);border-bottom:1px solid var(--border);background:var(--panel2);display:flex;justify-content:space-between;align-items:center;gap:8px}
  .panel .body{padding:10px 12px;overflow:auto;flex:1;min-height:0}
  /* explicit placement — deterministic, no auto-flow overflow */
  #p-prompt{grid-area:prompt} #p-trace{grid-area:trace} #p-scores{grid-area:scores}
  #p-artifact{grid-area:artifact} #p-judge{grid-area:judge}
  #p-prompt pre{font-size:13px;line-height:1.55}
  pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:11.5px;line-height:1.45;color:#cdd3df}
  .kv{display:flex;justify-content:space-between;gap:10px;padding:3px 0;border-bottom:1px dashed var(--border)}
  .kv .k{color:var(--dim)} .kv .v{font-variant-numeric:tabular-nums}
  .score-hero{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
  .score-hero .num{font-size:34px;font-weight:800;line-height:1}
  .score-hero .den{color:var(--dim);font-size:15px}
  .comp{font-family:var(--mono);font-size:12px;color:var(--dim);margin:6px 0 12px}
  .comp b{color:var(--fg)}
  .dim{margin:8px 0}
  .dim .row{display:flex;align-items:center;gap:8px}
  .dim .nm{flex:0 0 42%;font-size:12px}
  .bar{flex:1;height:8px;background:var(--panel2);border-radius:5px;overflow:hidden}
  .bar > i{display:block;height:100%;background:linear-gradient(90deg,#3fb950,#6ea8fe)}
  .dim .sc{font-variant-numeric:tabular-nums;color:var(--dim);flex:0 0 34px;text-align:right}
  .dim .reason{color:var(--dim);font-size:11px;margin:3px 0 0 2px;line-height:1.4}
  table.hard{width:100%;border-collapse:collapse;font-size:11.5px}
  table.hard td{padding:4px 6px;border-bottom:1px solid var(--border);vertical-align:top}
  table.hard td.st{width:18px} .ok{color:var(--good)} .no{color:var(--bad)}
  .faithrow{padding:6px 0;border-bottom:1px solid var(--border);font-size:11.5px}
  .faithrow a{color:var(--accent);text-decoration:none;word-break:break-all}
  .muted{color:var(--dim)}
  details{margin:4px 0} summary{cursor:pointer;color:var(--accent);font-size:11px}
  .tag{display:inline-block;font-family:var(--mono);font-size:10px;padding:1px 5px;border-radius:4px;background:var(--panel2);border:1px solid var(--border);color:var(--dim);margin-left:6px}
  .sep{height:1px;background:var(--border);margin:10px 0}
  .traceseg{color:var(--warn);font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;margin:2px 0 6px;text-transform:uppercase}
</style></head>
<body>
<div id="topbar">
  <span class="nav" id="first" title="First (Home)">⏮</span>
  <span class="nav" id="prev" title="←">◀</span>
  <span id="counter"></span>
  <span class="nav" id="next" title="→">▶</span>
  <span class="nav" id="last" title="Last (End)">⏭</span>
  <span id="taskname"></span>
  <span class="badge" id="passbadge"></span>
  <span class="chip" id="scorechip"></span>
  <span class="chip" id="deltachip"></span>
  <span class="grow"></span>
  <span class="chip" id="ttchip"></span>
  <span class="chip" id="jsonchip"></span>
  <select id="jump"></select>
  <span class="hint">← → navigate · <span id="runname">__RUNNAME__</span></span>
</div>
<div id="board">
  <div class="panel" id="p-prompt"><h2>Full Prompt <span id="promptlen" class="tag"></span></h2><div class="body"><pre id="prompt"></pre></div></div>
  <div class="panel" id="p-scores"><h2>Scores</h2><div class="body" id="scores"></div></div>
  <div class="panel" id="p-artifact"><h2>Artifact <span id="artmeta" class="tag"></span></h2><div class="body" id="artifact"></div></div>
  <div class="panel" id="p-trace"><h2>Agent trace <span class="tag">compact — exactly what the effectiveness judge sees</span></h2><div class="body" id="trace"></div></div>
  <div class="panel" id="p-judge"><h2>Judge trace</h2><div class="body" id="judge"></div></div>
</div>
<script>
const DATA = __DATA__;
const R = DATA.records;
let i = 0;
const esc = s => (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmt = n => (n==null||isNaN(n))?'–':(Math.round(n*100)/100).toFixed(2);

// build jump menu
const jump = document.getElementById('jump');
R.forEach((r,idx)=>{ const o=document.createElement('option'); o.value=idx;
  o.textContent = `${idx+1}. ${r.task}  (${fmt(r.final_score)})`; jump.appendChild(o); });
jump.addEventListener('change',()=>{ i=+jump.value; render(); });

function dimHTML(dims){
  if(!dims||!dims.length) return '<div class="muted">no dimension scores</div>';
  return dims.map(d=>{
    const s = (typeof d.score==='number')? d.score : null;
    const pct = s==null?0:Math.max(0,Math.min(100,(s/5)*100));
    return `<div class="dim"><div class="row"><span class="nm">${esc(d.name)}</span>
      <span class="bar"><i style="width:${pct}%"></i></span>
      <span class="sc">${s==null?'–':s}/5</span></div>
      ${d.reason?`<div class="reason">${esc(d.reason)}</div>`:''}</div>`;
  }).join('');
}

function render(){
  const r = R[i];
  document.getElementById('counter').textContent = `${i+1} / ${R.length}`;
  document.getElementById('taskname').textContent = r.task;
  document.getElementById('taskname').title = r.task;
  jump.value = i;
  const pb = document.getElementById('passbadge');
  pb.textContent = r.passed?'PASS':'FAIL'; pb.className = 'badge '+(r.passed?'pass':'fail');
  document.getElementById('scorechip').textContent = 'score '+fmt(r.final_score)+' / 5';
  const dc = document.getElementById('deltachip');
  if(r.two_phase_score!=null){ const dl = r.final_score - r.two_phase_score;
    dc.textContent = 'Δ2-phase '+(dl>=0?'+':'')+fmt(dl); dc.style.display='';
    dc.style.color = dl>=0?'var(--good)':'var(--bad)';
  } else dc.style.display='none';
  const tt=document.getElementById('ttchip');
  tt.textContent='two-turn '+(r.two_turn_fired?'fired':'no'); tt.className='chip '+(r.two_turn_fired?'on':'off');
  const jc=document.getElementById('jsonchip');
  jc.textContent='JSON '+(r.json_parsed?'ok':'✗')+(r.summary_json_source?(' ('+r.summary_json_source+')'):'');
  jc.className='chip '+(r.json_parsed?'on':'off');

  // prompt
  document.getElementById('prompt').textContent = r.prompt || '(no prompt file found)';
  document.getElementById('promptlen').textContent = (r.prompt||'').length+' chars';

  // scores
  const c = r.components||{};
  let sh = `<div class="score-hero"><span class="num" style="color:${r.passed?'var(--good)':'var(--bad)'}">${fmt(r.final_score)}</span><span class="den">/ 5 · threshold 3.0</span></div>`;
  sh += `<div class="comp">= <b>${fmt(c.llm_overall)}</b> llm − <b>${fmt(c.hard_penalty)}</b> hard-penalty + <b>${fmt(c.faithfulness_bonus)}</b> faith-bonus</div>`;
  if(r.regrade_note) sh += `<div class="reason" style="color:var(--warn);margin-bottom:8px">⟳ ${esc(r.regrade_note)}</div>`;
  sh += `<div style="font-size:11px;color:var(--dim);margin-bottom:4px">EFFECTIVENESS DIMENSIONS</div>`+dimHTML(r.dimensions);
  sh += `<div class="sep"></div><div style="font-size:11px;color:var(--dim);margin-bottom:4px">HARD CONSTRAINTS · pass-rate ${fmt((r.hard_pass_rate||0)*100)}%</div>`;
  if(r.hard_report.length){
    sh += '<table class="hard">'+r.hard_report.map(h=>`<tr><td class="st ${h.passed?'ok':'no'}">${h.passed?'✓':'✗'}</td><td><b>${esc(h.name)}</b><br><span class="muted">${esc(h.detail)}</span></td></tr>`).join('')+'</table>';
  } else sh += '<div class="muted">none</div>';
  const f=r.faith||{};
  sh += `<div class="sep"></div><div style="font-size:11px;color:var(--dim)">FAITHFULNESS (URL grounding, bonus-only)</div>
    <div class="kv"><span class="k">fields matched</span><span class="v">${f.matched??'–'} / ${f.total??'–'}</span></div>
    <div class="kv"><span class="k">pages fetched</span><span class="v">${f.fetched??'–'}</span></div>
    <div class="kv"><span class="k">score /5</span><span class="v">${fmt(f.score_5)}</span></div>`;
  document.getElementById('scores').innerHTML = sh;

  // artifact
  let art = '';
  if(r.summary_json){ art += `<div class="traceseg">summary.json (structured answer)</div><pre>${esc(JSON.stringify(r.summary_json,null,2))}</pre>`; }
  if(r.artifacts && r.artifacts.length){
    art += r.artifacts.map(a=>`<details><summary>📄 ${esc(a.name)} (${a.body.length} chars)</summary><pre>${esc(a.body)}</pre></details>`).join('');
  }
  document.getElementById('artifact').innerHTML = art || '<div class="muted">no artifact</div>';
  const nart = (r.artifacts?r.artifacts.length:0) + (r.summary_json?1:0);
  document.getElementById('artmeta').textContent = nart+' item'+(nart===1?'':'s');

  // trace (compact digest the judge sees)
  let tr = `<div class="traceseg">Agent Final Response — head [:8000]</div><pre>${esc(r.trace_head)||'<span class=muted>(empty)</span>'}</pre>`;
  tr += `<div class="sep"></div><div class="traceseg">Agent stdout / trace — tail [-12000:] (of ${r.trace_full_len} chars)</div><pre>${esc(r.trace_tail)||'<span class=muted>(no stdout captured)</span>'}</pre>`;
  document.getElementById('trace').innerHTML = tr;

  // judge trace
  let jg = '';
  if(r.evidence_summary) jg += `<div class="traceseg">evidence summary</div><div class="reason" style="font-size:12px">${esc(r.evidence_summary)}</div><div class="sep"></div>`;
  jg += `<div class="traceseg">per-dimension reasoning</div>`+dimHTML(r.dimensions);
  if(r.dimensions_prior && r.dimensions_prior.length){
    const changed = JSON.stringify(r.dimensions_prior.map(d=>d.score))!==JSON.stringify(r.dimensions.map(d=>d.score));
    if(changed) jg += `<details><summary>prior (pre-faith-revision) scores</summary>${dimHTML(r.dimensions_prior)}</details>`;
  }
  if(r.faith_details && r.faith_details.length){
    jg += `<div class="sep"></div><div class="traceseg">faithfulness per-URL</div>`;
    jg += r.faith_details.map(fd=>`<div class="faithrow">
      <div>${fd.fetched?'<span class=ok>✓ fetched</span>':'<span class=no>✗ not fetched</span>'} ${fd.http_status?('· HTTP '+fd.http_status):''}</div>
      <a href="${esc(fd.url)}" target="_blank">${esc(fd.url)}</a>
      ${fd.claim?`<div class="muted" style="margin-top:3px">${esc(fd.claim)}</div>`:''}
      ${fd.detail?`<details><summary>detail</summary><pre>${esc(typeof fd.detail==='string'?fd.detail:JSON.stringify(fd.detail,null,2))}</pre></details>`:''}
    </div>`).join('');
  }
  if(r.feedback){ jg += `<div class="sep"></div><details><summary>raw feedback.txt</summary><pre>${esc(r.feedback)}</pre></details>`; }
  document.getElementById('judge').innerHTML = jg;

  // reset scroll of panels
  document.querySelectorAll('.body').forEach(b=>b.scrollTop=0);
}
function go(d){ i=(i+d+R.length)%R.length; render(); }
document.getElementById('prev').onclick=()=>go(-1);
document.getElementById('next').onclick=()=>go(1);
document.getElementById('first').onclick=()=>{i=0;render();};
document.getElementById('last').onclick=()=>{i=R.length-1;render();};
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT') return;
  if(e.key==='ArrowLeft'){go(-1);e.preventDefault();}
  else if(e.key==='ArrowRight'){go(1);e.preventDefault();}
  else if(e.key==='Home'){i=0;render();}
  else if(e.key==='End'){i=R.length-1;render();}
});
render();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
