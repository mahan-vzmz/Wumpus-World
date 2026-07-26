"""Generate the README poster SVG for the interactive demo.

Renders a real frame of the Wumpus Lair episode (agent's belief map mid-run,
with genuine reasoning-log lines) so the poster is an honest preview of the
demo, in the same visual language as the demo page itself.
"""

from __future__ import annotations

from pathlib import Path

from wumpus.agents.rule_agent import RuleAgent
from wumpus.viz.recorder import record_episode_from_file

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "data" / "maps" / "holdout_suite" / "03_wumpus_hazard_map_01.txt"
OUTPUT_PATH = ROOT / "docs" / "assets" / "demo_preview.svg"

FRAME_INDEX = 9  # mid-run: one confirmed Wumpus, one survived pit, live suspicions

CELL_FILL = {
    "u": ("#10141d", "#1c2230", None),
    "s": ("#12362b", "#1f7a58", None),
    "v": ("#1b2431", "#2c3648", None),
    "p": ("#3a2f14", "#8a6a1f", "🕳"),
    "w": ("#3a1c1c", "#8a3a3a", "👹"),
    "b": ("#37281a", "#8a5a2a", "⚠"),
    "P": ("#5c4310", "#dfa32b", "🕳"),
    "W": ("#571f1f", "#e05252", "👹"),
    "k": ("#1b2431", "#dfa32b", "🕳"),
    "x": ("#232833", "#3a4150", None),
}

LOG_PICKS = (
    ("STENCH at", "#ef8f8f"),
    ("= POSSIBLE_WUMPUS", "#ef8f8f"),
    ("KNOWN_PIT", "#e3b158"),
    ("= CONFIRMED_WUMPUS", "#ff8383"),
    ("P1 EMERGENCY RETREAT", "#ff8383"),
    ("P2 EXPLORE safe frontier", "#6fd7de"),
)


def _pick_log_lines(record: dict) -> list[tuple[str, str]]:
    """First genuine occurrence of each interesting trace kind, in order."""
    lines: list[tuple[str, str]] = []
    for needle, color in LOG_PICKS:
        for frame in record["frames"]:
            hit = next(
                (
                    ln
                    for ln in frame["trace"]
                    if needle in ln and not ln.strip().startswith("NO_")
                ),
                None,
            )
            if hit:
                lines.append((hit.strip(), color))
                break
    return lines


def build_poster_svg(record: dict) -> str:
    frame = record["frames"][min(FRAME_INDEX, len(record["frames"]) - 1)]
    belief = frame["belief"]
    exit_r, exit_c = record["exit"]
    agent_r, agent_c = frame["pos"]

    width, height = 940, 470
    cell, gap = 42, 4
    bx, by = 42, 84  # board origin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">',
        '<title id="t">Interactive demo preview: the agent&#39;s live belief map</title>',
        f'<rect width="{width}" height="{height}" rx="18" fill="#0b0e15"/>',
        # header
        '<text x="42" y="46" font-family="Consolas, monospace" font-size="24" font-weight="700" '
        'fill="#edf1f8">WUMPUS WORLD <tspan fill="#41d3dc">/ inside the mind of an AI</tspan></text>',
        '<text x="42" y="68" font-family="Consolas, monospace" font-size="12" fill="#5c6577">'
        "live belief map · reasoning log · x-ray truth reveal — one self-contained HTML file</text>",
        # board plate
        f'<rect x="{bx - 10}" y="{by - 10}" width="{8 * (cell + gap) - gap + 20}" '
        f'height="{8 * (cell + gap) - gap + 20}" rx="12" fill="#0f131d" stroke="#1d2430"/>',
    ]

    for r in range(8):
        for c in range(8):
            code = belief[r * 8 + c]
            fill, stroke, glyph = CELL_FILL[code]
            x = bx + c * (cell + gap)
            y = by + r * (cell + gap)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            )
            if glyph:
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 6}" text-anchor="middle" '
                    f'font-size="18">{glyph}</text>'
                )
    # exit + agent markers
    ex = bx + exit_c * (cell + gap)
    ey = by + exit_r * (cell + gap)
    parts.append(
        f'<text x="{ex + cell / 2}" y="{ey + cell / 2 + 6}" text-anchor="middle" font-size="16">🚪</text>'
    )
    ax = bx + agent_c * (cell + gap)
    ay = by + agent_r * (cell + gap)
    parts.extend(
        [
            f'<circle cx="{ax + cell / 2}" cy="{ay + cell / 2}" r="{cell / 2 - 3}" '
            'fill="none" stroke="#41d3dc" stroke-width="2" opacity="0.85"/>',
            f'<text x="{ax + cell / 2}" y="{ay + cell / 2 + 7}" text-anchor="middle" font-size="20">🤖</text>',
        ]
    )

    # right column: mind log excerpt (real lines)
    lx = 470
    parts.extend(
        [
            f'<text x="{lx}" y="104" font-family="Consolas, monospace" font-size="13" '
            'fill="#5c6577" letter-spacing="3">MIND LOG · LIVE REASONING</text>',
            f'<rect x="{lx - 14}" y="118" width="{width - lx - 28}" height="196" rx="12" '
            'fill="#131826" stroke="#1d2430"/>',
        ]
    )
    y = 148
    for text, color in _pick_log_lines(record):
        shown = text if len(text) <= 52 else text[:49] + "…"
        parts.append(
            f'<text x="{lx}" y="{y}" font-family="Consolas, monospace" font-size="12.5" '
            f'fill="{color}">{shown.replace("&", "&amp;").replace("<", "&lt;")}</text>'
        )
        y += 28

    # legend + call to action
    parts.extend(
        [
            f'<text x="{lx}" y="352" font-family="Consolas, monospace" font-size="12" fill="#98a2b4">'
            '<tspan fill="#1f7a58">■</tspan> safe · <tspan fill="#8a6a1f">■</tspan> pit? · '
            '<tspan fill="#8a3a3a">■</tspan> wumpus? · <tspan fill="#dfa32b">■</tspan> confirmed · '
            "🤖 agent · 🚪 exit</text>",
            f'<rect x="{lx - 2}" y="376" width="300" height="44" rx="10" fill="#0f2b2d" '
            'stroke="#41d3dc" stroke-width="1.5"/>',
            f'<text x="{lx + 148}" y="404" text-anchor="middle" font-family="Consolas, monospace" '
            'font-size="15" font-weight="700" fill="#41d3dc">▶ OPEN THE INTERACTIVE DEMO</text>',
            f'<text x="{lx}" y="446" font-family="Consolas, monospace" font-size="11" fill="#5c6577">'
            "python -m wumpus visualize → docs/demo/index.html</text>",
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    record = record_episode_from_file(MAP_PATH, RuleAgent(), seed=42)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_poster_svg(record), encoding="utf-8")
    print(f"Generated {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
