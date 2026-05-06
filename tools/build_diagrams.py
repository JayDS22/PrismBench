"""
Generate the architecture and roadmap diagrams for the report.

The architecture diagram is rendered with Graphviz/dot for a clean
Mermaid/Flow.io-style visual. The roadmap is matplotlib because Gantt
charts are not Graphviz's strength.

Outputs:
    figures/architecture.png  - Graphviz flow diagram, horizontal
    figures/roadmap.png       - matplotlib three-lane research roadmap

Run with:
    .venv/bin/python tools/build_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

NAVY = "#1F3A8A"
NAVY_LIGHT = "#E8EDF8"
NAVY_DEEP = "#13256B"
GREY = "#666666"
GREY_LIGHT = "#F4F4F4"
GREEN = "#2E7D5B"
GREEN_LIGHT = "#E2F0EA"
ORANGE = "#C97A1F"
ORANGE_LIGHT = "#FBEBD9"
ACCENT_OK = GREEN
ACCENT_WIP = ORANGE
ACCENT_PLAN = "#7B7B7B"


def build_architecture():
    """Render a Mermaid-style flow diagram via Graphviz/dot.

    Left-right rankdir with size=10x5 gives a roughly 2:1 aspect
    that fits Letter page width well at moderate max-height.
    """
    dot = Digraph(
        "PrismBench",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "splines": "spline",
            "nodesep": "0.55",
            "ranksep": "0.45",
            "bgcolor": "white",
            "fontname": "Helvetica",
            "label": "PrismBench System Architecture",
            "labelloc": "t",
            "fontsize": "16",
            "fontcolor": NAVY,
            "pad": "0.30",
            "dpi": "160",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "11",
            "margin": "0.18,0.10",
            "penwidth": "1.4",
        },
        edge_attr={
            "color": NAVY,
            "penwidth": "1.4",
            "arrowsize": "0.8",
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": NAVY,
        },
    )

    # 1. Configs cluster (top, side-by-side)
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(
            label="Configs",
            style="rounded,dashed",
            color=GREY,
            fontcolor=GREY,
            fontsize="11",
            penwidth="1.0",
            margin="14",
            rank="same",
        )
        c.node("tasks", "tasks.yaml",
               shape="note", fillcolor=GREY_LIGHT, color=GREY,
               fontcolor="#222")
        c.node("datasets", "datasets.yaml",
               shape="note", fillcolor=GREY_LIGHT, color=GREY,
               fontcolor="#222")
        c.node("agents_yaml", "agents.yaml",
               shape="note", fillcolor=GREY_LIGHT, color=GREY,
               fontcolor="#222")

    # 2. Orchestrator
    dot.node(
        "orchestrator",
        "<<B>Orchestrator</B><BR/>"
        "<FONT POINT-SIZE='11'>task_runner.py</FONT><BR/>"
        "<FONT POINT-SIZE='10' COLOR='#444'>"
        "stages data &middot; wraps CodeCarbon<BR/>"
        "dispatches (agent, task, run)"
        "</FONT>>",
        fillcolor=NAVY_LIGHT, color=NAVY, fontcolor="#111",
    )

    # 3. Agent runs cluster (three categories side by side)
    with dot.subgraph(name="cluster_agents") as c:
        c.attr(
            label="Agent execution  (3 protocols, 165 cells)",
            style="rounded,filled",
            fillcolor="#F8FAFE",
            color=NAVY,
            fontcolor=NAVY,
            fontsize="11",
            penwidth="1.0",
            rank="same",
            margin="14",
        )
        c.node(
            "automl",
            "<<B>AutoML</B><BR/>"
            "<FONT POINT-SIZE='10' COLOR='#444'>"
            "autogluon, pycaret"
            "</FONT>>",
            fillcolor="#FFFFFF", color=NAVY,
        )
        c.node(
            "agentic",
            "<<B>Agentic LLM</B><BR/>"
            "<FONT POINT-SIZE='10' COLOR='#444'>"
            "claude_code, chatgpt_ada,<BR/>"
            "claude_api_raw, langgraph"
            "</FONT>>",
            fillcolor="#FFFFFF", color=NAVY,
        )
        c.node(
            "paradigm",
            "<<B>Paradigm-specific</B><BR/>"
            "<FONT POINT-SIZE='10' COLOR='#444'>"
            "autogen, smolagents, pandasai"
            "</FONT>>",
            fillcolor="#FFFFFF", color=NAVY,
        )

    # 4. Scoring
    dot.node(
        "evaluator",
        "<<B>Six-dim scoring</B><BR/>"
        "<FONT POINT-SIZE='11'>evaluator.py + llm_judge.py</FONT><BR/>"
        "<FONT POINT-SIZE='10' COLOR='#444'>"
        "D1 sklearn &middot; D2 pylint + LLM<BR/>"
        "D3 SHAP + LLM &middot; D4 / D5 / D6"
        "</FONT>>",
        fillcolor=NAVY_LIGHT, color=NAVY,
    )

    # 5. Scorecard storage
    dot.node(
        "scorecard",
        "<<B>Scorecard</B><BR/>"
        "<FONT POINT-SIZE='10' COLOR='#444'>"
        "master_scorecard.csv<BR/>"
        "summary.csv"
        "</FONT>>",
        shape="cylinder",
        fillcolor=GREEN_LIGHT, color=GREEN, fontcolor="#111",
    )

    # 6. Router
    dot.node(
        "router",
        "<<B>PrismBench Router</B><BR/>"
        "<FONT POINT-SIZE='11'>router.py</FONT><BR/>"
        "<FONT POINT-SIZE='10' COLOR='#444'>"
        "WSM &middot; TOPSIS &middot; PROMETHEE-II<BR/>"
        "sensitivity sweep, Pareto suite"
        "</FONT>>",
        fillcolor=ORANGE_LIGHT, color=ORANGE, fontcolor="#111",
    )

    # 7. Outputs cluster (bottom)
    with dot.subgraph(name="cluster_outputs") as c:
        c.attr(
            label="Outputs",
            style="rounded,dashed",
            color=GREY,
            fontcolor=GREY,
            fontsize="11",
            penwidth="1.0",
            rank="same",
            margin="14",
        )
        c.node("ranks", "Top-1 agent\nper preset",
               shape="note", fillcolor="#FFFFFF", color=GREY)
        c.node("figures_n", "figures/\npareto_*, sensitivity_*",
               shape="note", fillcolor="#FFFFFF", color=GREY)
        c.node("stats", "Friedman / Nemenyi\nWilcoxon",
               shape="note", fillcolor="#FFFFFF", color=GREY)

    # Edges
    for src in ("tasks", "datasets", "agents_yaml"):
        dot.edge(src, "orchestrator")
    dot.edge("orchestrator", "automl")
    dot.edge("orchestrator", "agentic")
    dot.edge("orchestrator", "paradigm")
    dot.edge("automl", "evaluator")
    dot.edge("agentic", "evaluator")
    dot.edge("paradigm", "evaluator")
    dot.edge("evaluator", "scorecard")
    dot.edge("scorecard", "router")
    dot.edge("router", "ranks")
    dot.edge("router", "figures_n")
    dot.edge("router", "stats")

    out_base = FIGURES / "architecture"
    dot.render(filename=str(out_base), cleanup=True, format="png")
    rendered = out_base.with_suffix(".png")
    print(f"wrote {rendered}")


def lane_bar(ax, y, x_start, x_end, color, label):
    """Bar with the label drawn ABOVE the bar."""
    height = 0.30
    rect = FancyBboxPatch(
        (x_start, y - height / 2), x_end - x_start, height,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=color, edgecolor="white", linewidth=1.2,
        alpha=0.95,
    )
    ax.add_patch(rect)
    ax.text(
        (x_start + x_end) / 2, y + height / 2 + 0.08, label,
        ha="center", va="bottom",
        fontsize=9, color="#222", fontweight="bold",
    )


def status_tag(ax, x, y, color, text):
    ax.text(
        x, y, text,
        ha="left", va="center",
        fontsize=9, color=color, fontweight="bold", style="italic",
    )


def build_roadmap():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-0.8, 6.6)
    ax.axis("off")

    ax.text(
        7.0, 6.30,
        "PrismBench Research Roadmap",
        ha="center", va="top",
        fontsize=15, fontweight="bold", color=NAVY,
    )
    ax.text(
        7.0, 5.95,
        "Snapshot today (May 2026); paper drafting in flight; productionization scoped through 2027.",
        ha="center", va="top",
        fontsize=10, color=GREY, style="italic",
    )

    lane_label_w = 2.20
    timeline_x0 = lane_label_w + 0.10
    timeline_w = 14.0 - timeline_x0 - 0.10

    def t(q):
        return timeline_x0 + (q + 0.5) * (timeline_w / 7)

    quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026",
                "Q1 2027", "Q2 2027", "Q3+ 2027"]
    for i, label in enumerate(quarters):
        x = t(i)
        ax.text(x, 5.05, label, ha="center", va="center",
                fontsize=9, color=GREY, fontweight="bold")
        ax.plot([x, x], [-0.10, 4.80], color="#dddddd",
                linewidth=0.6, linestyle="--", zorder=0)

    ax.plot([timeline_x0, 14.0 - 0.10], [4.80, 4.80],
            color="#999999", linewidth=0.8)

    today_x = t(1) + 0.40 * (timeline_w / 7)
    ax.plot([today_x, today_x], [-0.10, 4.80],
            color="#C0392B", linewidth=1.3, alpha=0.6)
    ax.text(today_x, 4.95, "today", ha="center", va="bottom",
            fontsize=8.5, color="#C0392B", fontweight="bold", style="italic")

    lane_y = {"benchmark": 3.85, "paper": 2.55, "prod": 1.25}
    for name, label in [
        ("benchmark", "1.  Benchmark\n     artifact"),
        ("paper", "2.  Research\n     paper"),
        ("prod", "3.  Productionization\n     and extensions"),
    ]:
        ax.text(
            0.05, lane_y[name], label,
            ha="left", va="center", fontsize=10, color=NAVY,
            fontweight="bold",
        )

    qstep = timeline_w / 7

    def bar_q(q_start, q_end):
        return (timeline_x0 + q_start * qstep,
                timeline_x0 + q_end * qstep)

    s, e = bar_q(0.05, 0.85)
    lane_bar(ax, lane_y["benchmark"], s, e, ACCENT_OK, "smoke + data")
    s, e = bar_q(0.95, 1.55)
    lane_bar(ax, lane_y["benchmark"], s, e, ACCENT_OK, "165 cells")
    s, e = bar_q(1.65, 1.95)
    lane_bar(ax, lane_y["benchmark"], s, e, ACCENT_OK, "report")
    status_tag(ax, e + 0.10, lane_y["benchmark"], ACCENT_OK, "DONE")

    s, e = bar_q(1.40, 2.20)
    lane_bar(ax, lane_y["paper"], s, e, ACCENT_WIP, "drafting")
    s, e = bar_q(2.25, 3.30)
    lane_bar(ax, lane_y["paper"], s, e, ACCENT_WIP, "causal ablation")
    s, e = bar_q(3.35, 4.20)
    lane_bar(ax, lane_y["paper"], s, e, ACCENT_WIP, "submit")
    status_tag(ax, e + 0.10, lane_y["paper"], ACCENT_WIP, "IN PROGRESS")

    s, e = bar_q(3.50, 4.80)
    lane_bar(ax, lane_y["prod"], s, e, ACCENT_PLAN, "live leaderboard")
    s, e = bar_q(4.85, 6.00)
    lane_bar(ax, lane_y["prod"], s, e, ACCENT_PLAN, "cross-modal")
    s, e = bar_q(6.05, 6.55)
    lane_bar(ax, lane_y["prod"], s, e, ACCENT_PLAN, "v2")
    status_tag(ax, e + 0.10, lane_y["prod"], ACCENT_PLAN, "PLANNED")

    legend_y = 0.20
    legend_items = [
        ("Done (snapshot artifact)", ACCENT_OK),
        ("In progress (paper draft)", ACCENT_WIP),
        ("Planned (productionization)", ACCENT_PLAN),
    ]
    cursor = 2.5
    for label, color in legend_items:
        sw = FancyBboxPatch(
            (cursor, legend_y - 0.10), 0.32, 0.22,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=color, edgecolor="white", linewidth=0,
        )
        ax.add_patch(sw)
        ax.text(
            cursor + 0.42, legend_y, label,
            ha="left", va="center", fontsize=9.5, color="#333",
        )
        cursor += 3.50

    out = FIGURES / "roadmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    build_architecture()
    build_roadmap()
