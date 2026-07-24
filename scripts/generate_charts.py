"""Generate README-ready SVG charts from committed benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "benchmark_summary.json"
OUTPUT_PATH = ROOT / "docs" / "assets" / "benchmark_win_rate.svg"


def build_win_rate_svg(summary: dict[str, dict[str, object]]) -> str:
    order = ["search", "rules", "greedy", "ml", "random"]
    labels = {
        "search": "A* Search (full visibility)",
        "rules": "Rule-based",
        "greedy": "Greedy baseline",
        "ml": "Random Forest",
        "random": "Random baseline",
    }
    colors = {
        "search": "#2563eb",
        "rules": "#0f766e",
        "greedy": "#d97706",
        "ml": "#7c3aed",
        "random": "#64748b",
    }

    width, height = 900, 430
    chart_left, chart_right = 245, 840
    chart_width = chart_right - chart_left
    bar_height, row_gap = 38, 24
    first_y = 105

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Holdout win rate by agent</title>',
        '<desc id="desc">Win rate on twenty independent holdout maps.</desc>',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        '<text x="42" y="45" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#0f172a">Holdout win rate by agent</text>',
        '<text x="42" y="73" font-family="Arial, sans-serif" font-size="14" fill="#475569">20 independent maps · training seeds 100–199 · holdout seeds 2000–2019</text>',
    ]

    for tick in range(0, 101, 20):
        x = chart_left + chart_width * tick / 100
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="370" stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="392" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{tick}%</text>',
            ]
        )

    for index, agent in enumerate(order):
        value = float(summary[agent]["win_rate_pct"])
        y = first_y + index * (bar_height + row_gap)
        bar_width = chart_width * value / 100
        if value >= 15:
            value_x = chart_left + bar_width - 10
            value_anchor = "end"
            value_color = "white"
        else:
            value_x = chart_left + bar_width + 10
            value_anchor = "start"
            value_color = "#334155"
        parts.extend(
            [
                f'<text x="225" y="{y + 25}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" font-weight="600" fill="#1e293b">{labels[agent]}</text>',
                f'<rect x="{chart_left}" y="{y}" width="{chart_width}" height="{bar_height}" rx="7" fill="#e2e8f0"/>',
                f'<rect x="{chart_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="7" fill="{colors[agent]}"/>',
                f'<text x="{value_x:.1f}" y="{y + 25}" text-anchor="{value_anchor}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="{value_color}">{value:.0f}%</text>',
            ]
        )

    parts.extend(
        [
            '<text x="42" y="414" font-family="Arial, sans-serif" font-size="12" fill="#64748b">A* is an upper bound and is not information-equivalent to partial-observation agents.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_win_rate_svg(payload["summary"]),
        encoding="utf-8",
    )
    print(f"Generated {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
