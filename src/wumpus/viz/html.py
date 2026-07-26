"""Standalone HTML demo builder.

Curates a set of maps (easy → very hard), records **every agent** on each of
them with :mod:`wumpus.viz.recorder`, and injects the JSON payload into a
single self-contained HTML page — no server, no build step, no network.

The player shows, per map, an agent switcher ("MIND"): the rule-based
reasoner with its live belief map and reasoning log, the full-visibility A*
planner (X-ray on by default — it sees everything), the ML imitator with its
own belief map, and the greedy/random baselines with a visited trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.viz.recorder import record_episode_from_file

DEFAULT_MODEL_PATH = Path("artifacts/models/random_forest.joblib")

#: Fixed switcher order; "rules" first — it is the star of the demo.
AGENT_ORDER: tuple[str, ...] = ("rules", "search", "ml", "greedy", "random")

#: agent -> (display label, visibility)
AGENT_META: dict[str, tuple[str, str]] = {
    "rules": ("RULE-BASED", "fog"),
    "search": ("A* SEARCH", "full"),
    "ml": ("ML FOREST", "fog"),
    "greedy": ("GREEDY", "fog"),
    "random": ("RANDOM", "fog"),
}


@dataclass(frozen=True)
class DemoEpisodeSpec:
    """One curated map tab of the demo."""

    map_path: str  # repo-relative
    episode_id: str
    title: str
    stars: int  # 1..5 difficulty
    tagline: str
    fatal: bool = False  # the honest-failure showcase tab


#: Curated line-up, easy → very hard, closing on an honest failure case.
CURATED_EPISODES: tuple[DemoEpisodeSpec, ...] = (
    DemoEpisodeSpec(
        "data/maps/holdout_suite/01_easy_map_01.txt",
        "first-steps", "First Steps", 1,
        "An open field. Watch the fog give way to inference.",
    ),
    DemoEpisodeSpec(
        "data/maps/holdout_suite/02_pit_heavy_map_03.txt",
        "pit-field", "Pit Field", 2,
        "Four hidden pits. One breeze at a time.",
    ),
    DemoEpisodeSpec(
        "data/maps/holdout_suite/03_wumpus_hazard_map_01.txt",
        "wumpus-lair", "Wumpus Lair", 3,
        "Two Wumpuses in the dark. One wrong step is fatal.",
    ),
    DemoEpisodeSpec(
        "data/maps/holdout_suite/04_gold_hunter_map_04.txt",
        "gold-rush", "Gold Rush", 4,
        "Treasure worth a detour — if the detour is safe.",
    ),
    DemoEpisodeSpec(
        "data/maps/holdout_suite/05_hard_complex_map_04.txt",
        "gauntlet", "The Gauntlet", 5,
        "Pits, a Wumpus, walls, and a tight health budget.",
    ),
    DemoEpisodeSpec(
        "data/maps/holdout_suite/05_hard_complex_map_03.txt",
        "last-stand", "Last Stand", 5,
        "The map that broke the reasoner. Can any mind escape?",
        fatal=True,
    ),
)

_SHARED_KEYS = ("truth", "exit", "config", "grid_size", "map_name")


def _make_agent(name: str, model_path: Path | None):
    if name == "rules":
        return RuleAgent()
    if name == "search":
        return SearchAgent()
    if name == "ml":
        return MLAgent(model_path=model_path or DEFAULT_MODEL_PATH)
    if name == "greedy":
        return GreedyExitAgent()
    if name == "random":
        return RandomAgent()
    raise ValueError(f"unknown demo agent '{name}'")


def build_demo_payload(
    repo_root: Path,
    seed: int = 42,
    model_path: Path | None = None,
    include_ml: bool = True,
    specs: tuple[DemoEpisodeSpec, ...] = CURATED_EPISODES,
) -> dict[str, Any]:
    """Record every agent on every curated map and assemble the payload."""
    agent_names = [a for a in AGENT_ORDER if include_ml or a != "ml"]
    if include_ml:
        resolved_model = model_path or (repo_root / DEFAULT_MODEL_PATH)
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"ML model not found at '{resolved_model}'. Run "
                "'python -m wumpus train' first, or build the demo without ML."
            )
    else:
        resolved_model = None

    episodes: list[dict[str, Any]] = []
    for spec in specs:
        runs: dict[str, dict[str, Any]] = {}
        shared: dict[str, Any] | None = None
        for name in agent_names:
            record = record_episode_from_file(
                repo_root / spec.map_path,
                _make_agent(name, resolved_model),
                seed=seed,
                agent_name=name,
            )
            current_shared = {k: record[k] for k in _SHARED_KEYS}
            if shared is None:
                shared = current_shared
            else:
                assert shared == current_shared, "map/config must match across agents"
            label, visibility = AGENT_META[name]
            run: dict[str, Any] = {
                "agent": name,
                "label": label,
                "visibility": visibility,
                "seed": seed,
                "result": record["result"],
                "frames": record["frames"],
            }
            if "planner" in record:
                run["planner"] = record["planner"]
            runs[name] = run

        assert shared is not None
        episodes.append(
            {
                "id": spec.episode_id,
                "title": spec.title,
                "stars": spec.stars,
                "tagline": spec.tagline,
                "fatal": spec.fatal,
                **shared,
                "runs": runs,
            }
        )

    return {
        "generator": "python -m wumpus visualize",
        "agents": agent_names,
        "seed": seed,
        "episodes": episodes,
    }


def build_demo_html(payload: dict[str, Any]) -> str:
    """Render the payload into the self-contained demo page."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA_JSON__", data)


def write_demo(
    repo_root: Path,
    output: Path,
    seed: int = 42,
    model_path: Path | None = None,
    include_ml: bool = True,
) -> int:
    """Build the demo and write it to ``output``; returns bytes written."""
    payload = build_demo_payload(
        repo_root, seed=seed, model_path=model_path, include_ml=include_ml
    )
    html = build_demo_html(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))


# ---------------------------------------------------------------------------
# The page template. Single dark theme by design: the whole premise is fog.
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wumpus World — Inside the Mind of an AI</title>
<style>
  :root {
    --bg: #0b0e15;
    --panel: #131826;
    --panel-2: #0f131d;
    --ring: rgba(255,255,255,.07);
    --ink: #edf1f8;
    --ink-2: #98a2b4;
    --ink-3: #5c6577;
    --cyan: #41d3dc;
    --teal: #22b07e;
    --amber: #dfa32b;
    --red: #e66;
    --gold: #f2c14e;
    --good: #3ac96a;
    --bad: #e05252;
    --mono: "Cascadia Code", "JetBrains Mono", Consolas, "SF Mono", ui-monospace, monospace;
    --sans: "Segoe UI Variable Display", "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; }
  html, body { height: 100%; }
  body {
    background:
      radial-gradient(1100px 500px at 75% -10%, rgba(65,211,220,.07), transparent 60%),
      radial-gradient(900px 500px at 10% 110%, rgba(223,163,43,.05), transparent 55%),
      var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.45;
    padding: clamp(12px, 2.5vw, 28px);
    max-width: 1180px;
    margin: 0 auto;
  }
  a { color: var(--cyan); text-decoration: none; }
  a:hover { text-decoration: underline; }
  kbd {
    font-family: var(--mono); font-size: .72em; padding: 1px 6px;
    border: 1px solid var(--ring); border-bottom-width: 2px; border-radius: 5px;
    background: var(--panel-2); color: var(--ink-2);
  }

  /* ---------- header ---------- */
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .brand { font-family: var(--mono); font-size: clamp(19px, 2.6vw, 26px); font-weight: 700; letter-spacing: .04em; }
  .brand .dot { color: var(--cyan); }
  .brand .sub { color: var(--ink-3); font-weight: 400; font-size: .62em; letter-spacing: .12em; text-transform: uppercase; }
  .hdr-meta { font-family: var(--mono); font-size: 12px; color: var(--ink-3); display: flex; gap: 14px; align-items: center; }
  .chip {
    border: 1px solid var(--ring); border-radius: 999px; padding: 3px 10px;
    color: var(--cyan); background: rgba(65,211,220,.06); letter-spacing: .08em;
  }
  .lede { color: var(--ink-2); max-width: 68ch; margin: 10px 0 18px; font-size: 14.5px; }
  .lede em { color: var(--ink); font-style: normal; border-bottom: 1px dashed var(--ink-3); }

  /* ---------- tabs + agent switcher ---------- */
  .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .tab {
    font-family: var(--mono); font-size: 12.5px; letter-spacing: .02em;
    background: var(--panel-2); color: var(--ink-2);
    border: 1px solid var(--ring); border-radius: 9px;
    padding: 7px 12px; cursor: pointer; transition: all .18s ease;
  }
  .tab:hover { color: var(--ink); border-color: rgba(255,255,255,.16); }
  .tab.active { color: var(--ink); background: var(--panel); border-color: rgba(65,211,220,.45); box-shadow: 0 0 0 1px rgba(65,211,220,.25), 0 4px 14px rgba(0,0,0,.35); }
  .tab .stars { color: var(--gold); margin-left: 7px; letter-spacing: -1px; }
  .tab .skull { margin-left: 7px; filter: saturate(.7); }

  .agents { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
  .agents-lbl { font-family: var(--mono); font-size: 10.5px; letter-spacing: .22em; color: var(--ink-3); }
  .seg { display: inline-flex; border: 1px solid var(--ring); border-radius: 10px; overflow: hidden; background: var(--panel-2); }
  .seg button {
    font-family: var(--mono); font-size: 11.5px; letter-spacing: .04em;
    color: var(--ink-2); background: transparent; border: 0; border-right: 1px solid var(--ring);
    padding: 7px 12px; cursor: pointer; transition: all .15s ease;
  }
  .seg button:last-child { border-right: 0; }
  .seg button:hover { color: var(--ink); }
  .seg button.active { color: #06282b; background: var(--cyan); font-weight: 700; }
  .vis-chip { font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em;
              border: 1px solid var(--ring); border-radius: 999px; padding: 3px 10px; }
  .vis-chip.fog  { color: var(--cyan); border-color: rgba(65,211,220,.45); background: rgba(65,211,220,.07); }
  .vis-chip.full { color: var(--gold); border-color: rgba(242,193,78,.5); background: rgba(242,193,78,.09); }
  .agent-note { font-size: 12.5px; color: var(--ink-3); margin: 0 0 14px; max-width: 80ch; }

  /* ---------- stage ---------- */
  .stage { display: grid; grid-template-columns: minmax(320px, 560px) minmax(300px, 1fr); gap: 18px; align-items: start; }
  @media (max-width: 860px) { .stage { grid-template-columns: 1fr; } }

  .board-card, .log-card {
    background: linear-gradient(180deg, rgba(255,255,255,.02), transparent 40%), var(--panel);
    border: 1px solid var(--ring); border-radius: 16px;
    box-shadow: 0 18px 40px rgba(0,0,0,.35);
  }
  .board-card { padding: 14px; position: relative; overflow: hidden; }
  .board-head { display: flex; justify-content: space-between; align-items: baseline; margin: 2px 4px 10px; }
  .board-title { font-weight: 650; font-size: 15px; letter-spacing: .01em; }
  .board-tag { color: var(--ink-3); font-size: 12px; }

  .board {
    position: relative; display: grid;
    grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr);
    gap: 4px; aspect-ratio: 1 / 1;
    background: var(--panel-2); border-radius: 12px; padding: 8px;
    border: 1px solid rgba(255,255,255,.05);
  }
  .cell {
    position: relative; border-radius: 7px; background: #0e1219;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
    transition: background .45s ease, box-shadow .45s ease;
    display: grid; place-items: center;
  }
  .glyph { font-size: clamp(13px, 2.6vw, 21px); line-height: 1; pointer-events: none;
           opacity: 0; transform: scale(.6); transition: opacity .35s ease, transform .35s ease, filter .35s ease; }

  /* belief layers */
  .b-s { background: rgba(34,176,126,.13); box-shadow: inset 0 0 0 1px rgba(34,176,126,.28); }
  .b-v { background: #1b2431; box-shadow: inset 0 0 0 1px rgba(255,255,255,.06); }
  .b-v::after { content: ""; position: absolute; width: 4px; height: 4px; border-radius: 50%;
                background: rgba(255,255,255,.16); }
  .b-p { background: rgba(223,163,43,.15); box-shadow: inset 0 0 0 1px rgba(223,163,43,.35); }
  .b-w { background: rgba(230,102,102,.13); box-shadow: inset 0 0 0 1px rgba(230,102,102,.35); }
  .b-b { background: linear-gradient(135deg, rgba(223,163,43,.16) 50%, rgba(230,102,102,.16) 50%);
         box-shadow: inset 0 0 0 1px rgba(223,140,60,.4); }
  .b-p .glyph, .b-w .glyph, .b-b .glyph { opacity: .5; transform: scale(.86); filter: grayscale(.5); }
  .b-P { background: rgba(223,163,43,.30); box-shadow: inset 0 0 0 2px rgba(223,163,43,.8); }
  .b-W { background: rgba(224,82,82,.28);  box-shadow: inset 0 0 0 2px rgba(224,82,82,.85); }
  .b-P .glyph, .b-W .glyph { opacity: 1; transform: scale(1); filter: none; animation: pop .5s ease; }
  .b-k { background: #1b2431; box-shadow: inset 0 0 0 2px rgba(223,163,43,.55); }
  .b-k .glyph { opacity: .85; transform: scale(.95); }
  .b-x { background: repeating-linear-gradient(45deg, #232833 0 6px, #1a1f29 6px 12px);
         box-shadow: inset 0 0 0 1px rgba(255,255,255,.09); }
  @keyframes pop { 0% { transform: scale(.4); } 55% { transform: scale(1.25); } 100% { transform: scale(1); } }

  /* truth (x-ray) layer */
  .truth { position: absolute; inset: 0; display: grid; place-items: center;
           font-size: clamp(13px, 2.6vw, 21px); opacity: 0; transform: scale(.55);
           transition: opacity .4s ease, transform .4s ease; pointer-events: none; }
  body.xray .truth { opacity: 1; transform: scale(1); }
  body.xray .cell { background: #0d1118 !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04) !important; }
  body.xray .cell .glyph { opacity: 0 !important; }
  body.xray .cell.t-wall { background: repeating-linear-gradient(45deg, #2a2f3a 0 6px, #20242e 6px 12px) !important; }
  .exit-mark { position: absolute; inset: 0; display: grid; place-items: center; pointer-events: none;
               font-size: clamp(11px, 2.1vw, 17px); opacity: .95;
               filter: drop-shadow(0 0 6px rgba(242,193,78,.45)); }
  .cell.is-exit { box-shadow: inset 0 0 0 1px rgba(242,193,78,.4); }

  /* agent orb */
  #agent {
    position: absolute; display: grid; place-items: center;
    font-size: clamp(15px, 3vw, 24px); z-index: 5; pointer-events: none;
    filter: drop-shadow(0 0 9px rgba(65,211,220,.65));
    transition: transform .3s cubic-bezier(.25,.8,.3,1);
  }
  #agent .ring { position: absolute; inset: 12%; border-radius: 50%;
                 border: 1.5px solid rgba(65,211,220,.55); animation: breathe 2.2s ease-in-out infinite; }
  @keyframes breathe { 0%,100% { transform: scale(.92); opacity: .55; } 50% { transform: scale(1.08); opacity: 1; } }

  /* board overlays */
  .stamp {
    position: absolute; inset: 0; display: grid; place-items: center; z-index: 8; pointer-events: none;
    opacity: 0; transition: opacity .4s ease;
  }
  .stamp span {
    font-family: var(--mono); font-size: clamp(26px, 5vw, 44px); font-weight: 800; letter-spacing: .18em;
    padding: 8px 26px; border: 3px solid currentColor; border-radius: 10px;
    transform: rotate(-7deg) scale(1.4); transition: transform .35s cubic-bezier(.2,1.6,.4,1);
    background: rgba(11,14,21,.72); backdrop-filter: blur(2px);
  }
  .stamp.show { opacity: 1; }
  .stamp.show span { transform: rotate(-7deg) scale(1); }
  .stamp.win span { color: var(--good); text-shadow: 0 0 22px rgba(58,201,106,.5); }
  .stamp.loss span { color: var(--bad); text-shadow: 0 0 22px rgba(224,82,82,.5); }
  .stamp.stall span { color: var(--amber); text-shadow: 0 0 22px rgba(223,163,43,.45); }
  .flash { position: absolute; inset: 0; border-radius: 12px; pointer-events: none; opacity: 0; z-index: 6; }
  .flash.hit { animation: hitflash .55s ease; }
  @keyframes hitflash { 0% { opacity: 0; } 25% { opacity: 1; box-shadow: inset 0 0 60px 18px rgba(224,82,82,.55); } 100% { opacity: 0; } }
  .spark { position: absolute; z-index: 7; pointer-events: none; font-size: 13px;
           animation: fly 1s ease-out forwards; }
  @keyframes fly { to { transform: translate(var(--dx), var(--dy)) rotate(240deg); opacity: 0; } }

  /* instruments */
  .instruments { display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
                 margin-top: 12px; font-family: var(--mono); font-size: 12.5px; }
  .meter { flex: 1 1 150px; min-width: 140px; }
  .meter .lbl { color: var(--ink-3); font-size: 10.5px; letter-spacing: .14em; margin-bottom: 4px; }
  .bar { height: 8px; border-radius: 99px; background: #0d1119; border: 1px solid var(--ring); overflow: hidden; }
  .bar i { display: block; height: 100%; width: 100%;
           background: linear-gradient(90deg, #2fd07f, #7ddc57); border-radius: 99px;
           transition: width .3s ease, background .3s ease; }
  .bar.mid i  { background: linear-gradient(90deg, #dfa32b, #e8c34e); }
  .bar.low i  { background: linear-gradient(90deg, #e05252, #e67); }
  .stat b { color: var(--ink); font-weight: 650; font-variant-numeric: tabular-nums; }
  .stat { color: var(--ink-3); letter-spacing: .05em; }
  .badges { display: flex; gap: 8px; margin-left: auto; }
  .badge { border: 1px solid var(--ring); border-radius: 7px; padding: 3px 9px;
           color: var(--ink-3); font-size: 11px; letter-spacing: .1em; transition: all .25s ease; }
  .badge.on.breeze { color: #ffd479; border-color: rgba(223,163,43,.6); background: rgba(223,163,43,.12);
                     box-shadow: 0 0 12px rgba(223,163,43,.25); }
  .badge.on.stench { color: #ff9c9c; border-color: rgba(224,82,82,.6); background: rgba(224,82,82,.12);
                     box-shadow: 0 0 12px rgba(224,82,82,.25); }
  .status-chip { border-radius: 7px; padding: 3px 10px; font-size: 11px; letter-spacing: .12em;
                 border: 1px solid var(--ring); color: var(--cyan); background: rgba(65,211,220,.07); }
  .status-chip.won  { color: var(--good); border-color: rgba(58,201,106,.5); background: rgba(58,201,106,.1); }
  .status-chip.dead { color: var(--bad); border-color: rgba(224,82,82,.5); background: rgba(224,82,82,.1); }
  .status-chip.stall { color: var(--amber); border-color: rgba(223,163,43,.5); background: rgba(223,163,43,.1); }

  /* ---------- mind log ---------- */
  .log-card { display: flex; flex-direction: column; min-height: 420px; max-height: 660px; }
  .log-head { display: flex; justify-content: space-between; align-items: center;
              padding: 12px 16px; border-bottom: 1px solid var(--ring);
              font-family: var(--mono); font-size: 12px; letter-spacing: .16em; color: var(--ink-3); }
  .log-head b { color: var(--cyan); letter-spacing: .05em; }
  .log { flex: 1; overflow-y: auto; padding: 10px 14px 14px; font-family: var(--mono);
         font-size: 12px; scroll-behavior: smooth; }
  .log::-webkit-scrollbar { width: 8px; } .log::-webkit-scrollbar-thumb { background: #232a38; border-radius: 8px; }
  .entry { border-left: 2px solid transparent; padding: 6px 10px; margin-bottom: 6px; border-radius: 0 8px 8px 0; opacity: .58; }
  .entry.current { border-left-color: var(--cyan); background: rgba(65,211,220,.05); opacity: 1; }
  .entry.about { opacity: 1; border-left-color: var(--gold); background: rgba(242,193,78,.05); }
  .entry-head { color: var(--ink-3); font-size: 10.5px; letter-spacing: .14em; margin-bottom: 3px; }
  .entry-head .act { color: var(--cyan); font-weight: 700; }
  .ln { white-space: pre-wrap; word-break: break-word; color: var(--ink-2); padding: 1px 0; }
  .ln.safe    { color: #58c9a0; }
  .ln.pit     { color: #e3b158; }
  .ln.wumpus  { color: #ef8f8f; }
  .ln.confirm { font-weight: 700; }
  .ln.policy  { color: #6fd7de; }
  .ln.danger  { color: #ff8383; font-weight: 700; }
  .ln.muted   { color: var(--ink-3); }
  .ln.gold    { color: #f2c14e; }
  .legend { border-top: 1px solid var(--ring); padding: 10px 16px 13px;
            display: flex; flex-wrap: wrap; gap: 8px 14px; font-size: 11px; color: var(--ink-3); }
  .legend .k { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 12px; height: 12px; border-radius: 4px; display: inline-block; }

  /* ---------- control deck ---------- */
  .deck {
    position: sticky; bottom: 10px; margin-top: 18px; z-index: 20;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    background: rgba(15,19,29,.88); backdrop-filter: blur(10px);
    border: 1px solid var(--ring); border-radius: 14px; padding: 10px 14px;
    box-shadow: 0 14px 34px rgba(0,0,0,.45);
  }
  .btn {
    font-family: var(--mono); font-size: 13px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--ring); border-radius: 9px;
    min-width: 38px; height: 34px; padding: 0 10px; cursor: pointer; transition: all .15s ease;
  }
  .btn:hover { border-color: rgba(65,211,220,.5); color: var(--cyan); }
  .btn.primary { background: rgba(65,211,220,.14); border-color: rgba(65,211,220,.55); color: var(--cyan); font-weight: 700; }
  .btn.toggled { background: rgba(242,193,78,.15); border-color: rgba(242,193,78,.6); color: var(--gold); }
  .scrub { flex: 1 1 200px; display: flex; align-items: center; gap: 10px; min-width: 180px; }
  .scrub input[type=range] { flex: 1; accent-color: var(--cyan); }
  .frame-lbl { font-family: var(--mono); font-size: 11.5px; color: var(--ink-3); min-width: 74px; text-align: right;
               font-variant-numeric: tabular-nums; }
  select {
    font-family: var(--mono); font-size: 12px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--ring); border-radius: 8px; height: 34px; padding: 0 8px;
  }
  .hints { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); display: flex; gap: 10px; }
  @media (max-width: 700px) { .hints { display: none; } }

  footer.credits { margin-top: 14px; text-align: center; color: var(--ink-3); font-size: 11.5px; font-family: var(--mono); }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
  }
</style>
</head>
<body>

<header>
  <div class="brand"><span class="dot">◆</span> WUMPUS WORLD <span class="sub">/ inside the mind of an AI</span></div>
  <div class="hdr-meta">
    <span class="chip" id="agentChip">MIND · RULE-BASED</span>
    <a href="https://github.com/mahan-vzmz/Wumpus-World" target="_blank" rel="noopener">source ↗</a>
  </div>
</header>

<p class="lede">
  Five minds, one hidden dungeon. Most of them have <em>never seen this map</em> — pits and Wumpuses
  hide in the dark, and all they feel is a breeze or a stench next door. Every coloured square is an
  <em>inference</em>, not vision. Switch minds below, and hit <kbd>X</kbd> to compare belief with truth.
</p>

<nav class="tabs" id="tabs"></nav>

<div class="agents">
  <span class="agents-lbl">MIND</span>
  <div class="seg" id="agentSeg"></div>
  <span class="vis-chip fog" id="visChip">FOG OF WAR</span>
</div>
<p class="agent-note" id="agentNote"></p>

<main class="stage">
  <section class="board-card">
    <div class="board-head">
      <div class="board-title" id="epTitle"></div>
      <div class="board-tag" id="epTag"></div>
    </div>
    <div class="board" id="board"></div>
    <div class="flash" id="flash"></div>
    <div class="stamp" id="stamp"><span id="stampText"></span></div>
    <div class="instruments">
      <div class="meter">
        <div class="lbl">HEALTH</div>
        <div class="bar" id="healthBar"><i id="healthFill"></i></div>
      </div>
      <span class="stat">STEP <b id="stSteps">0</b></span>
      <span class="stat">GOLD <b id="stGold">0</b></span>
      <span class="stat">SCORE <b id="stScore">0</b></span>
      <div class="badges">
        <span class="badge breeze" id="bBreeze">BREEZE</span>
        <span class="badge stench" id="bStench">STENCH</span>
        <span class="status-chip" id="statusChip">EXPLORING</span>
      </div>
    </div>
  </section>

  <aside class="log-card">
    <div class="log-head"><span><b>MIND LOG</b> · live reasoning</span><span id="logStep"></span></div>
    <div class="log" id="log"></div>
    <div class="legend">
      <span class="k"><span class="sw" style="background:#0e1219;box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)"></span>unknown</span>
      <span class="k"><span class="sw" style="background:rgba(34,176,126,.35)"></span>inferred safe</span>
      <span class="k"><span class="sw" style="background:#232d3c"></span>visited</span>
      <span class="k"><span class="sw" style="background:rgba(223,163,43,.5)"></span>pit?</span>
      <span class="k"><span class="sw" style="background:rgba(230,102,102,.5)"></span>wumpus?</span>
      <span class="k"><span class="sw" style="background:rgba(223,163,43,.95)"></span>confirmed</span>
      <span class="k">🕳 pit</span><span class="k">👹 wumpus</span><span class="k">🪙 gold</span><span class="k">🚪 exit</span>
    </div>
  </aside>
</main>

<footer class="deck">
  <button class="btn" id="btnRestart" title="Restart (R)">⟲</button>
  <button class="btn" id="btnPrev" title="Step back (←)">◀</button>
  <button class="btn primary" id="btnPlay" title="Play / pause (Space)">▶</button>
  <button class="btn" id="btnNext" title="Step forward (→)">▶▶</button>
  <div class="scrub">
    <input type="range" id="scrub" min="0" max="1" value="0" step="1" aria-label="Frame">
    <span class="frame-lbl" id="frameLbl">0 / 0</span>
  </div>
  <select id="speed" title="Playback speed">
    <option value="1.6">0.5×</option>
    <option value="0.8" selected>1×</option>
    <option value="0.4">2×</option>
    <option value="0.2">4×</option>
  </select>
  <button class="btn" id="btnXray" title="X-ray: reveal hidden truth (X)">👁 X-RAY</button>
  <span class="hints"><span><kbd>Space</kbd> play</span><span><kbd>←</kbd><kbd>→</kbd> step</span><span><kbd>X</kbd> x-ray</span></span>
</footer>

<footer class="credits">
  built from real episode traces · deterministic seed · no libraries, one file ·
  <span id="credEp"></span>
</footer>

<script>
"use strict";
const DATA = __DATA_JSON__;

/* ---------- state ---------- */
const S = { ep: 0, agent: "rules", frame: 0, playing: false, timer: null, xray: false, autoXrayDone: false };
const $ = (id) => document.getElementById(id);
const board = $("board"), logEl = $("log");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

const BELIEF_TITLE = { u:"unknown", s:"inferred SAFE", v:"visited", p:"possible pit",
  w:"possible Wumpus", b:"possible pit or Wumpus", P:"CONFIRMED pit", W:"CONFIRMED Wumpus",
  k:"known pit (survived it)", x:"wall" };
const BELIEF_GLYPH = { p:"🕳", w:"👹", b:"⚠", P:"🕳", W:"👹", k:"🕳" };
const TRUTH_GLYPH = { P:"🕳", W:"👹", G:"🪙" };

const AGENT_INFO = {
  rules:  { note: "Logical inference under fog — persistent negative facts, hazard confirmation, risk-ranked exploration. The star of this demo." },
  search: { note: "Sees the entire map and plans the optimal route with A* before its first move — the expert upper bound, not a fair rival. X-ray is on because nothing is hidden from it." },
  ml:     { note: "A Random Forest imitating the A* expert from 397 observable features. It keeps the same belief map as the reasoner; illegal and known-deadly moves are masked." },
  greedy: { note: "Always steps toward the exit, ignoring every warning. No memory, no fear — watch the pit count." },
  random: { note: "Uniform random legal moves. The floor every other mind is measured against." },
};

function ep() { return DATA.episodes[S.ep]; }
function run() { return ep().runs[S.agent] || ep().runs[Object.keys(ep().runs)[0]]; }

/* belief for frame i — agents without a KB get a visited-trail reconstruction */
function beliefAt(i) {
  const frames = run().frames;
  if (frames[0].belief) return frames[Math.min(i, frames.length - 1)].belief;
  const cells = new Array(64).fill("u");
  for (let k = 0; k <= i && k < frames.length; k++) cells[frames[k].pos[0] * 8 + frames[k].pos[1]] = "v";
  return cells.join("");
}

/* ---------- build board for current episode ---------- */
let cells = [], agentEl = null;
function buildBoard() {
  const e = ep();
  board.innerHTML = "";
  cells = [];
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    const glyph = document.createElement("span");
    glyph.className = "glyph";
    cell.appendChild(glyph);
    const t = e.truth[r][c];
    if (t !== "*") {
      const truth = document.createElement("span");
      truth.className = "truth";
      if (t === "D") { cell.classList.add("t-wall"); truth.textContent = ""; }
      else truth.textContent = TRUTH_GLYPH[t] || "";
      cell.appendChild(truth);
    }
    if (r === e.exit[0] && c === e.exit[1]) {
      cell.classList.add("is-exit");
      const em = document.createElement("span");
      em.className = "exit-mark"; em.textContent = "🚪";
      cell.appendChild(em);
    }
    board.appendChild(cell);
    cells.push(cell);
  }
  agentEl = document.createElement("div");
  agentEl.id = "agent";
  agentEl.innerHTML = '<span class="ring"></span>🤖';
  board.appendChild(agentEl);
  requestAnimationFrame(placeAgent);
}
function cellRect(i) {
  const b = board.getBoundingClientRect(), r = cells[i].getBoundingClientRect();
  return { x: r.left - b.left, y: r.top - b.top, w: r.width, h: r.height };
}
function placeAgent() {
  const f = run().frames[S.frame], i = f.pos[0] * 8 + f.pos[1], rc = cellRect(i);
  agentEl.style.width = rc.w + "px"; agentEl.style.height = rc.h + "px";
  agentEl.style.transform = `translate(${rc.x}px, ${rc.y}px)`;
}

/* ---------- render one frame ---------- */
function classify(line) {
  if (/(EMERGENCY|DEAD|RISKY)/.test(line)) return "danger";
  if (/CONFIRMED_PIT|KNOWN_PIT/.test(line)) return "pit confirm";
  if (/CONFIRMED_WUMPUS/.test(line)) return "wumpus confirm";
  if (/(POSSIBLE_PIT|BREEZE|breeze|pit)/.test(line)) return "pit";
  if (/(POSSIBLE_WUMPUS|STENCH|stench|[Ww]umpus)/.test(line)) return "wumpus";
  if (/(SAFE|not a pit|not a Wumpus)/.test(line)) return "safe";
  if (/^(P[1-9]|FOLLOW|FALLBACK|  STEP)/.test(line)) return "policy";
  return "muted";
}
function aboutEntry() {
  const r = run();
  const entry = document.createElement("div");
  entry.className = "entry about";
  const head = document.createElement("div");
  head.className = "entry-head";
  head.textContent = `MIND · ${r.label}`;
  entry.appendChild(head);
  const note = document.createElement("div");
  note.className = "ln muted";
  note.textContent = AGENT_INFO[r.agent]?.note || "";
  entry.appendChild(note);
  if (r.planner) {
    const p = document.createElement("div");
    p.className = "ln gold";
    p.textContent = r.planner.solved
      ? `PLAN · ${r.planner.plan_length} moves · predicted score ${r.planner.predicted_score} · ` +
        `${r.planner.expanded_nodes} nodes in ${r.planner.planning_time_ms} ms`
      : "PLAN · no safe route exists";
    entry.appendChild(p);
  }
  return entry;
}
function renderLog() {
  const frames = run().frames;
  logEl.innerHTML = "";
  logEl.appendChild(aboutEntry());
  for (let i = 0; i <= S.frame && i < frames.length; i++) {
    const f = frames[i];
    if (!f.trace.length && f.action === null) continue;
    const entry = document.createElement("div");
    entry.className = "entry" + (i === S.frame ? " current" : "");
    const head = document.createElement("div");
    head.className = "entry-head";
    head.innerHTML = `STEP ${String(i).padStart(2, "0")} <span class="act">→ ${f.action ?? "—"}</span>`;
    entry.appendChild(head);
    for (const line of f.trace) {
      const ln = document.createElement("div");
      ln.className = "ln " + classify(line);
      ln.textContent = line;
      entry.appendChild(ln);
    }
    logEl.appendChild(entry);
  }
  logEl.scrollTop = logEl.scrollHeight;
  $("logStep").textContent = `frame ${S.frame + 1}/${frames.length}`;
}
function statusLabel(status) {
  if (status === "WON") return ["ESCAPED", "won"];
  if (status.startsWith("DEAD")) return ["DIED", "dead"];
  if (status === "RUNNING") return ["EXPLORING", ""];
  if (status === "STEP_LIMIT") return ["OUT OF STEPS", "stall"];
  if (status === "NO_SOLUTION") return ["NO ROUTE", "stall"];
  return [status, "dead"];
}
function render() {
  const e = ep(), frames = run().frames, f = frames[S.frame];
  const belief = beliefAt(S.frame);

  for (let i = 0; i < 64; i++) {
    const cell = cells[i], code = belief[i];
    cell.className = cell.className.replace(/\bb-\S+/g, "").trim();
    if (code !== "u") cell.classList.add("b-" + code);
    const glyph = cell.firstChild;
    glyph.textContent = BELIEF_GLYPH[code] || "";
    const r = (i / 8) | 0, c = i % 8;
    cell.title = `(${r + 1},${c + 1}) · ${BELIEF_TITLE[code]}` + (S.xray ? ` · truth: ${e.truth[r][c]}` : "");
  }
  placeAgent();

  const hp = Math.max(0, f.health) / e.config.initial_health;
  $("healthFill").style.width = (hp * 100).toFixed(1) + "%";
  $("healthBar").className = "bar" + (hp <= .28 ? " low" : hp <= .55 ? " mid" : "");
  $("stSteps").textContent = f.steps;
  $("stGold").textContent = f.gold;
  $("stScore").textContent = f.score;
  $("bBreeze").classList.toggle("on", f.breeze);
  $("bStench").classList.toggle("on", f.stench);

  const [label, cls] = statusLabel(f.status);
  const chip = $("statusChip");
  chip.textContent = label;
  chip.className = "status-chip" + (cls ? " " + cls : "");

  $("scrub").value = S.frame;
  $("frameLbl").textContent = `${S.frame + 1} / ${frames.length}`;
  renderLog();

  // event moments
  if (S.frame > 0) {
    const prev = frames[S.frame - 1];
    if (prev.health - f.health > 1 && !reduced) {           // fell into a pit
      const fl = $("flash"); fl.classList.remove("hit"); void fl.offsetWidth; fl.classList.add("hit");
    }
  }
  const last = S.frame === frames.length - 1;
  const stamp = $("stamp");
  if (last && f.status !== "RUNNING") {
    const won = f.status === "WON";
    const stall = f.status === "STEP_LIMIT" || f.status === "NO_SOLUTION";
    $("stampText").textContent = won ? "ESCAPED" : stall ? "STALLED" : "DEFEATED";
    stamp.className = "stamp show " + (won ? "win" : stall ? "stall" : "loss");
    if (won && !reduced) confetti(f.pos);
    if (!S.autoXrayDone && !S.xray) { S.autoXrayDone = true; setTimeout(() => setXray(true), 650); }
  } else {
    stamp.className = "stamp";
  }
}

function confetti(pos) {
  const i = pos[0] * 8 + pos[1], rc = cellRect(i);
  for (let k = 0; k < 14; k++) {
    const s = document.createElement("span");
    s.className = "spark";
    s.textContent = ["✦", "✧", "★"][k % 3];
    s.style.left = rc.x + rc.w / 2 + "px"; s.style.top = rc.y + rc.h / 2 + "px";
    s.style.color = ["#f2c14e", "#41d3dc", "#3ac96a"][k % 3];
    s.style.setProperty("--dx", (Math.cos(k / 14 * 6.283) * (46 + (k % 4) * 16)) + "px");
    s.style.setProperty("--dy", (Math.sin(k / 14 * 6.283) * (46 + (k % 4) * 16) - 22) + "px");
    board.appendChild(s);
    setTimeout(() => s.remove(), 1050);
  }
}

/* ---------- x-ray ---------- */
function setXray(on) {
  S.xray = on;
  $("btnXray").classList.toggle("toggled", on);
  if (on && !reduced) {                                     // sonar sweep from the agent
    const f = run().frames[S.frame];
    for (let i = 0; i < 64; i++) {
      const truth = cells[i].querySelector(".truth");
      if (!truth) continue;
      const d = Math.hypot(((i / 8) | 0) - f.pos[0], (i % 8) - f.pos[1]);
      truth.style.transitionDelay = (d * 55) + "ms";
    }
  } else {
    for (const cell of cells) {
      const truth = cell.querySelector(".truth");
      if (truth) truth.style.transitionDelay = "0ms";
    }
  }
  document.body.classList.toggle("xray", on);
  render();
}

/* ---------- playback ---------- */
function setFrame(i, { fromUser = false } = {}) {
  S.frame = Math.max(0, Math.min(run().frames.length - 1, i));
  if (fromUser) stop();
  render();
}
function tick() {
  if (S.frame >= run().frames.length - 1) { stop(); return; }
  S.frame += 1; render();
  S.timer = setTimeout(tick, +$("speed").value * 1000);
}
function play() {
  if (S.playing) return;
  if (S.frame >= run().frames.length - 1) {
    S.frame = 0; S.autoXrayDone = false;
    setXray(run().visibility === "full");
  }
  S.playing = true; $("btnPlay").textContent = "❚❚";
  S.timer = setTimeout(tick, 120);
}
function stop() {
  S.playing = false; $("btnPlay").textContent = "▶";
  if (S.timer) { clearTimeout(S.timer); S.timer = null; }
}

/* ---------- agent + episode selection ---------- */
function applyRun({ autoplay = true } = {}) {
  const r = run();
  S.frame = 0; S.autoXrayDone = false;
  document.querySelectorAll("#agentSeg button").forEach(
    (b) => b.classList.toggle("active", b.dataset.agent === S.agent)
  );
  $("agentChip").textContent = `MIND · ${r.label}`;
  const vis = $("visChip");
  vis.textContent = r.visibility === "full" ? "FULL MAP" : "FOG OF WAR";
  vis.className = "vis-chip " + (r.visibility === "full" ? "full" : "fog");
  $("agentNote").textContent = AGENT_INFO[r.agent]?.note || "";
  $("credEp").textContent = `${ep().map_name} · mind: ${r.agent} · seed ${r.seed}`;
  $("scrub").max = r.frames.length - 1;
  setXray(r.visibility === "full");
  if (autoplay) setTimeout(play, 450);
}
function selectAgent(name) {
  if (!ep().runs[name]) return;
  stop();
  S.agent = name;
  applyRun();
}
function selectEpisode(idx, { autoplay = true } = {}) {
  stop();
  S.ep = idx;
  if (!ep().runs[S.agent]) S.agent = DATA.agents[0];
  const e = ep();
  document.querySelectorAll(".tab").forEach((t, i) => t.classList.toggle("active", i === idx));
  $("epTitle").textContent = e.title;
  $("epTag").textContent = e.tagline;
  history.replaceState(null, "", "#" + e.id);
  buildBoard();
  applyRun({ autoplay });
}
function buildTabs() {
  const tabs = $("tabs");
  DATA.episodes.forEach((e, i) => {
    const b = document.createElement("button");
    b.className = "tab";
    b.innerHTML = e.fatal
      ? `${e.title}<span class="skull">💀</span>`
      : `${e.title}<span class="stars">${"★".repeat(e.stars)}</span>`;
    b.addEventListener("click", () => selectEpisode(i));
    tabs.appendChild(b);
  });
}
function buildAgentSeg() {
  const seg = $("agentSeg");
  DATA.agents.forEach((name) => {
    const meta = DATA.episodes[0].runs[name];
    const b = document.createElement("button");
    b.dataset.agent = name;
    b.textContent = meta ? meta.label : name.toUpperCase();
    b.addEventListener("click", () => selectAgent(name));
    seg.appendChild(b);
  });
}

/* ---------- wiring ---------- */
buildTabs();
buildAgentSeg();
$("btnPlay").addEventListener("click", () => (S.playing ? stop() : play()));
$("btnPrev").addEventListener("click", () => setFrame(S.frame - 1, { fromUser: true }));
$("btnNext").addEventListener("click", () => setFrame(S.frame + 1, { fromUser: true }));
$("btnRestart").addEventListener("click", () => { S.autoXrayDone = false; setXray(run().visibility === "full"); setFrame(0, { fromUser: true }); play(); });
$("btnXray").addEventListener("click", () => setXray(!S.xray));
$("scrub").addEventListener("input", (ev) => setFrame(+ev.target.value, { fromUser: true }));
addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "SELECT") return;
  // preventDefault on the arrows matters: when the range scrubber has focus it
  // would otherwise ALSO step its own value natively, fire "input", and advance
  // a second frame — so one keypress skipped two frames.
  if (ev.code === "Space") { ev.preventDefault(); S.playing ? stop() : play(); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); setFrame(S.frame + 1, { fromUser: true }); }
  else if (ev.key === "ArrowLeft") { ev.preventDefault(); setFrame(S.frame - 1, { fromUser: true }); }
  else if (ev.key.toLowerCase() === "x") setXray(!S.xray);
  else if (ev.key.toLowerCase() === "r") { S.autoXrayDone = false; setXray(run().visibility === "full"); setFrame(0, { fromUser: true }); play(); }
});
addEventListener("resize", placeAgent);

const fromHash = DATA.episodes.findIndex((e) => "#" + e.id === location.hash);
selectEpisode(fromHash >= 0 ? fromHash : 0);
</script>
</body>
</html>
"""
