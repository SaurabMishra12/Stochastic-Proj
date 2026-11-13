"""Build the offline teaching report from completed, inspected experiment artifacts.

The generator never runs simulations and never invents missing results.  Its input
gate deliberately fails before replacing the report if required artifacts are absent.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import html
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, PythonLexer

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [
    ("overview", "A", "Project overview"),
    ("foundations", "B", "Mathematical foundations"),
    ("visual", "C", "Visual explanations"),
    ("code", "D", "Complete code walkthrough"),
    ("methods", "E", "Experiment methodology"),
    ("results", "F", "Executed results"),
    ("literature", "G", "Literature review"),
    ("discussion", "H", "Discussion and limitations"),
    ("conclusion", "I", "Conclusion and future work"),
    ("reproducibility", "J", "Reproducibility"),
]
REQUIRED_TABLES = ["combined_summary", "convergence_summary", "long_horizon_summary",
                   "sensitivity_summary", "synthetic_summary"]


def esc(value):
    """Escape text and attribute content before inserting it into the document."""
    return html.escape(str(value), quote=True)


def safe_json(value):
    """Encode embedded JSON without allowing an HTML script end tag."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")


def number(value, digits=4):
    """Format measured scalars while displaying unavailable estimates explicitly."""
    if value is None or value == "" or str(value).lower() in {"nan", "none", "null"}:
        return "not applicable"
    try:
        x = float(value)
        if not math.isfinite(x):
            return "not applicable"
        if x == 0:
            return "0"
        if abs(x) < .0001 or abs(x) >= 100000:
            return f"{x:.{digits-1}g}"
        if x.is_integer() and abs(x) >= 1:
            return f"{int(x):,}"
        return f"{x:.{digits}g}"
    except (TypeError, ValueError):
        return esc(value)


def read_csv(path):
    """Load complete trajectory-summary tables, preserving field names and nulls."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty or headerless required table: {path}")
        rows = []
        for source in reader:
            row = {}
            for key, value in source.items():
                if value is None or value.strip().lower() in {"", "nan", "none", "null"}:
                    row[key] = None
                    continue
                try:
                    x = float(value)
                    row[key] = (int(x) if x.is_integer() else x) if math.isfinite(x) else None
                except ValueError:
                    row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"Required table contains no executed rows: {path}")
    return rows


def load_inputs(root):
    """Validate the completed-run contract before reading report inputs."""
    metadata_path = root / "results/run_metadata/full_run.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Full experiments are not yet documented: results/run_metadata/full_run.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete" or not metadata.get("tests_passed"):
        raise ValueError("Report requires status=complete and passed preflight in full_run.json")
    tables = {name: read_csv(root / "results/tables" / f"{name}.csv") for name in REQUIRED_TABLES}
    grid = tables["combined_summary"]
    required = {"delta", "alpha", "n", "T", "dt", "phi_variance", "tail_velocity_mean"}
    if not required.issubset(grid[0]):
        raise ValueError(f"Combined summary lacks required fields: {sorted(required-set(grid[0]))}")
    if len(grid) != 72 or len({(r["delta"], r["alpha"]) for r in grid}) != 72:
        raise ValueError("Full grid must contain all 72 unique detuning/noise cells")
    if any(r["n"] < 500 for r in grid if r["alpha"] > 0):
        raise ValueError("A stochastic grid condition has fewer than 500 trajectories")
    synthetic=tables["synthetic_summary"]
    identities={(row["n_agents"],row["topology"],row["rho"],row["dropout"],row["delay"]) for row in synthetic}
    if len(synthetic)!=144 or len(identities)!=144 or any(row["n"]!=500 for row in synthetic):
        raise ValueError("Synthetic evidence requires all 144 unique cells with 500 complete networks per cell")
    synthetic_metadata=json.loads((root/"results/run_metadata/synthetic_run.json").read_text())
    if not synthetic_metadata.get("complete") or not synthetic_metadata.get("validation_complete"):
        raise ValueError("Synthetic execution and saved-output validation must both be complete")
    if len(tables["long_horizon_summary"])!=6 or any(row["n"]!=1000 or row["T"]!=200 for row in tables["long_horizon_summary"]):
        raise ValueError("All six 1000-path T=200 cohorts are required")
    if len(tables["sensitivity_summary"])!=18 or any(row["n"]!=500 for row in tables["sensitivity_summary"]):
        raise ValueError("All eighteen 500-path parameter-sensitivity cohorts are required")
    if len(tables["convergence_summary"])!=10:
        raise ValueError("Both five-mesh paired convergence scenarios are required")
    series_path = root / "results/tables/baseline_series.json"
    series = json.loads(series_path.read_text())
    for key in ["t", "deterministic_phi", "stochastic_phi", "mean_phi", "variance_phi", "mean_ci_low", "mean_ci_high"]:
        if key not in series or len(series[key]) != len(series["t"]):
            raise ValueError(f"Missing or misaligned measured baseline series: {key}")
    figures = sorted((root / "results/figures").glob("*.svg")) + sorted((root / "results/figures").glob("*.png"))
    if len(figures) < 8:
        raise ValueError("At least eight actual scientific figures are required")
    caption_path = root / "results/figures/captions.json"
    captions = json.loads(caption_path.read_text())
    for figure in figures:
        if figure.name not in captions:
            raise ValueError(f"Actual figure lacks its scientific caption: {figure.name}")
    vendor = root / "report/vendor/mathjax-3.2.2-tex-svg.js"
    if not vendor.exists() or vendor.stat().st_size < 1000000:
        raise FileNotFoundError("The offline MathJax SVG bundle is missing or truncated")
    return metadata, tables, series, figures, captions


def paragraph(text, class_name=""):
    """Wrap trusted authored HTML prose in a paragraph."""
    return f'<p class="{class_name}">{text}</p>'


def table(headers, rows, caption="", raw=False):
    """Render a labeled, horizontally contained table with escaped measured cells."""
    head = "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{str(x) if raw else esc(x)}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll" tabindex="0"><table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def equation(identifier, latex, title, symbols, intuition, implementation):
    """Place an equation beside its complete symbol key and implementation meaning."""
    return (f'<article class="equation" id="eq-{identifier}"><h4>{esc(title)}</h4>'
            f'<div class="math-display">\\[{latex}\\]</div>'
            + table(["Symbol", "Meaning and units"], symbols)
            + paragraph(f"<strong>Read it physically.</strong> {intuition}")
            + paragraph(f"<strong>In the code.</strong> {implementation}", "implementation") + "</article>")


def data_uri(path):
    """Embed local scientific images and source pages without file dependencies."""
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg"}.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported embedded image: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def svg_frame(title, content, height=280):
    """Give a conceptual SVG a readable title, common geometry and accessible text."""
    return (f'<svg class="concept-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" role="img" aria-label="{esc(title)}">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>'
            '<style>.s{stroke:currentColor;fill:none;stroke-width:2}.a{stroke:#007b73;fill:none;stroke-width:3}.d{stroke-dasharray:6 5}.box{fill:transparent;stroke:currentColor;stroke-width:1.6}text{fill:currentColor;font-family:Arial,sans-serif;font-size:17px}.small{font-size:14px}.accent{fill:#007b73}.line{stroke:currentColor;stroke-width:2;marker-end:url(#arrow)}</style>'
            + content + "</svg>")


def svg_box(x, y, width, height, lines):
    """Draw a labeled mechanism block rather than a decorative icon."""
    return f'<rect class="box" x="{x}" y="{y}" width="{width}" height="{height}" rx="3"/>' + "".join(
        f'<text x="{x+width/2}" y="{y+height/2+(i-(len(lines)-1)/2)*23+6}" text-anchor="middle">{esc(line)}</text>' for i, line in enumerate(lines))


def svg_arrow(x1, y1, x2, y2, label="", label_y=None):
    """Draw direction of signal or mathematical dependence with an optional label."""
    return (f'<line class="line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
            + (f'<text class="small" x="{(x1+x2)/2}" y="{label_y if label_y is not None else (y1+y2)/2-12}" text-anchor="middle">{esc(label)}</text>' if label else ""))


def diagrams():
    """Author all thirteen requested conceptual diagrams with explicit scope labels."""
    out = []
    def add(title, body, caption, height=280):
        out.append(f'<figure class="concept"><div class="figure-kicker">Conceptual diagram {len(out)+1:02d} / not simulation output</div>' + svg_frame(title, body, height) + f'<figcaption><strong>{esc(title)}.</strong> {caption}</figcaption></figure>')
    add("PLL architecture and feedback",
        svg_box(20,75,150,70,["Input carrier","phase θᵢ"])+svg_box(240,75,170,70,["Phase detector","sin(φ)"])+svg_box(490,75,155,70,["Loop filter","time scale a"])+svg_box(710,75,165,70,["Oscillator","phase θₒ"])+svg_arrow(170,110,240,110)+svg_arrow(410,110,490,110)+svg_arrow(645,110,710,110)+ '<path class="s" d="M790 145 V220 H325 V145" marker-end="url(#arrow)"/><text x="485" y="247" text-anchor="middle">Feedback: phase error φ = θᵢ − θₒ</text><text x="465" y="35" text-anchor="middle">Δ is persistent frequency mismatch; α dW perturbs the loop</text>',
        "The phase detector converts a periodic error into a corrective signal. Feedback tends to remove frequency mismatch; the filter creates an inertial phase-velocity state. This is a functional interpretation of the reduced model, not an identified hardware circuit.")
    add("Deterministic relaxation and stochastic fluctuation",
        '<path class="s" d="M60 40 V220 H840"/><path class="s d" d="M60 135 H840"/><path class="s" d="M65 55 C110 70 135 190 205 145 S290 126 350 135 S600 135 835 135"/><path class="a" d="M65 55 L95 66 L120 96 L145 146 L170 157 L200 143 L230 119 L260 140 L290 132 L320 151 L350 124 L380 142 L410 130 L440 150 L470 137 L500 123 L530 135 L560 149 L590 126 L620 139 L650 129 L680 146 L710 134 L740 122 L770 143 L800 131 L835 140"/><text x="60" y="25">Phase φ (rad)</text><text x="780" y="248">time</text><text x="615" y="110">stable phase φₛ</text><text x="90" y="255" class="small">Ink: α = 0; teal: α &gt; 0; shapes are illustrative</text>',
        "Damping removes a deterministic displacement. Noise continually perturbs velocity and therefore phase, leaving variability around a locked state. A fluctuating curve alone does not imply a completed slip.")
    add("Brownian paths are sums of independent increments",
        '<path class="s" d="M60 35 V205 H845"/><polyline class="a" points="65,150 140,120 215,138 290,95 365,108 440,60 515,85 590,115 665,72 740,96 830,48"/><line class="s d" x1="440" y1="60" x2="515" y2="60"/><line class="s" x1="515" y1="60" x2="515" y2="85"/><text x="480" y="45" text-anchor="middle">Δt</text><text x="530" y="79">ΔW</text><text x="80" y="25">W(t), piecewise-linear illustration</text><text x="120" y="245">ΔWₙ = √Δt Zₙ, Zₙ independent N(0,1), Var(ΔWₙ) = Δt</text>',
        "Brownian increments scale with the square root of elapsed time. The displayed polygon joins sampled values; the limiting Brownian path is continuous and almost surely nondifferentiable. Independent path streams differ from independent steps within a path.")
    add("Second-order formal equation to an Itô state system",
        svg_box(30,55,340,100,["a φ̈ + φ̇ + ab sin(φ)","= Δ − α ξ(t)"])+svg_box(520,45,350,130,["dφ = v dt","dv = [(Δ − v − ab sin φ)/a] dt","− (α/a) dW"])+svg_arrow(380,105,510,105,"v = φ̇")+'<text x="450" y="235" text-anchor="middle">Interpret ξ(t) dt as dW(t); white noise is not a pointwise ordinary function.</text>',
        "Introduce velocity, divide the acceleration balance by a, and interpret the white-noise integral as a Brownian increment. The phase coordinate has no direct diffusion term; noise enters through velocity.")
    add("One explicit Euler–Maruyama step",
        svg_box(20,75,155,85,["Old state","(φₙ, vₙ)"])+svg_box(245,20,235,70,["Evaluate drift","f(φₙ, vₙ) Δt"])+svg_box(245,150,235,70,["Generate ΔWₙ","g √Δt Zₙ"])+svg_box(560,75,305,85,["New state","Xₙ₊₁ = Xₙ + fₙ Δt + g ΔWₙ"])+svg_arrow(175,115,245,57)+svg_arrow(175,115,245,185)+svg_arrow(480,57,560,110)+svg_arrow(480,185,560,130)+'<text x="450" y="258" text-anchor="middle">Both coordinates use the same old state. Do not update v before computing φ.</text>',
        "The deterministic increment is proportional to Δt while the random increment is proportional to √Δt. Updating one coordinate and then evaluating the other drift would change the numerical method.")
    potential_points = " ".join(f"{x:.1f},{105-40*(-.15*((x-65)/110)-math.cos((x-65)/110)):.1f}" for x in range(65,836,5))
    add("Equilibria in a tilted periodic potential",
        '<path class="s" d="M60 25 V220 H850"/><polyline class="a" points="'+potential_points+'"/><circle class="accent" cx="81.6" cy="145.5" r="5"/><circle fill="none" stroke="currentColor" stroke-width="2" cx="394" cy="83.4" r="5"/><text x="65" y="22">U(φ) = −Δφ − ab cos(φ)</text><text x="100" y="245">Minima: stable wells; maxima: unstable barriers; tilt favors one direction</text>',
        "Local extrema obey sin(φ)=Δ/(ab). Stable branches have cos(φ)&gt;0; unstable branches have cos(φ)&lt;0. The sloping periodic potential is a mathematical landscape. It is not the complete two-dimensional basin boundary.")
    add("Locked wells disappear at the detuning threshold",
        svg_box(25,45,250,145,["|δ| < 1","Stable and unstable equilibria","Fluctuations and escapes"])+svg_box(325,45,250,145,["|δ| = 1","Equilibria coalesce","Nonhyperbolic threshold"])+svg_box(625,45,250,145,["|δ| > 1","No equilibrium","Persistent phase rotation"])+svg_arrow(275,115,325,115)+svg_arrow(575,115,625,115)+'<text x="450" y="245" text-anchor="middle">δ = Δ/(ab); coexistence of wells does not guarantee global acquisition in an inertial loop.</text>',
        "The algebraic existence threshold describes equilibria. Finite-horizon capture additionally depends on initial velocity, initial phase, damping and noise. Unlocked rotation is reported separately from locked-well slips.")
    add("Phase space carries both position and velocity",
        '<path class="s" d="M80 130 H830 M450 225 V30"/><path class="a" d="M170 65 C750 30 780 230 370 196 C190 180 215 65 525 75 C700 90 635 180 430 163 C330 155 360 115 485 121 C540 125 500 142 463 135"/><circle class="accent" cx="460" cy="133" r="5"/><text x="840" y="135">φ</text><text x="460" y="28">v</text><text x="500" y="210">Spiral shown only schematically</text><text x="95" y="255">At the same phase, different velocities can lead to different future wells.</text>',
        "The state is (φ,v), so a vertical line through a saddle phase is not generally the separatrix. The report's numerical saddle-manifold plot approximates the deterministic stable manifold and is kept separate from the operational dwell detector.")
    add("A completed phase slip requires arrival and persistence",
        '<path class="s" d="M65 40 V220 H840"/><path class="s d" d="M65 175 H840 M65 75 H840"/><rect x="65" y="60" width="775" height="30" fill="#007b73" opacity=".1"/><path class="a" d="M65 175 C180 160 200 80 250 160 S370 170 420 105 S510 60 565 75 L610 72 L665 78 L720 74 L780 77 L835 73"/><line class="s" x1="565" y1="32" x2="720" y2="32"/><text x="640" y="23" text-anchor="middle">continuous dwell ≥ τ</text><text x="95" y="145" class="small">excursion returns: no slip</text><text x="580" y="110" class="small">next-well core</text><text x="90" y="250">Stable centers differ by 2π rad; commitment time follows arrival by τ.</text>',
        "Crossing and returning does not count. A primary event commits only after entering the adjacent well's π/4 core and remaining for 0.5 time units. The event time is the commitment time, so waiting-time statistics include the dwell delay.")
    add("Monte Carlo uses independent complete paths",
        svg_box(20,85,180,85,["Recorded master seed","condition / path IDs"])+svg_box(285,20,205,55,["Brownian stream 1"])+svg_box(285,105,205,55,["Brownian stream 2"])+svg_box(285,190,205,55,["Brownian stream M"])+svg_box(595,75,275,110,["Independent path summaries","Means / Wilson intervals","Bootstrap rows / censoring"])+svg_arrow(200,125,285,48)+svg_arrow(200,125,285,133)+svg_arrow(200,125,285,218)+svg_arrow(490,48,595,105)+svg_arrow(490,133,595,130)+svg_arrow(490,218,595,155),
        "A trajectory is the sampling unit. Thousands of time samples from one path are correlated observations, not thousands of independent experiments. The same Brownian stream is deliberately shared only within paired time-step comparisons.")
    add("Experiment and evidence workflow",
        svg_box(20,75,140,95,["Source pages","Model derivation"])+svg_box(200,75,140,95,["Preflight","Tests pass"])+svg_box(380,75,140,95,["Full grid","Checkpoints"])+svg_box(560,75,140,95,["Independent-row","Analysis"])+svg_box(740,75,140,95,["Embedded report","Browser verify"])+svg_arrow(160,123,200,123)+svg_arrow(340,123,380,123)+svg_arrow(520,123,560,123)+svg_arrow(700,123,740,123)+'<text x="450" y="235" text-anchor="middle">Every measured table and figure must trace back to saved raw data and configuration.</text>',
        "The prerequisite tests protect the full experiment matrix. Checkpoints preserve expensive work; they are reusable only when source/configuration identity agrees. A report is generated from outputs after execution and cannot turn a planned condition into a completed result.")
    circles = "".join(f'<circle cx="{x}" cy="{y}" r="27" class="box"/><text x="{x}" y="{y+6}" text-anchor="middle">θ{i+1}</text>' for i,(x,y) in enumerate([(125,120),(300,55),(300,190)]))
    add("Synthetic oscillators, topology and coherence",
        circles+ '<path class="s" d="M150 110 L275 65 M150 130 L275 180 M300 83 V162"/>'+svg_box(475,65,370,120,["Coupling: sin(θⱼ(t−τ) − θᵢ(t−τ))","Noise: independent / correlated / shared","R = |mean exp(i θⱼ)|"])+svg_arrow(350,120,465,120)+'<text x="450" y="255" text-anchor="middle">R near 1: aligned phases; R near 0: dispersed phases. No claim about language reasoning.</text>',
        "Communication edges determine which phase differences contribute to coupling. Delay changes the phase sampled from a neighbor; dropout removes messages; shared noise can align motion without improving every notion of agreement. This is a separately specified Kuramoto-type model.")
    add("A gated LLM communication research proposal",
        svg_box(20,75,170,85,["Benchmark item","Gold hidden"])+svg_box(260,25,170,55,["Agent A"])+svg_box(260,110,170,55,["Agent B"])+svg_box(260,195,170,55,["Agent C"])+svg_box(510,65,190,130,["Measured disagreement","Candidate signal","Communication policy"])+svg_box(755,75,125,110,["Evaluator","Accuracy","Tokens/cost"])+svg_arrow(190,115,260,52)+svg_arrow(190,115,260,137)+svg_arrow(190,115,260,222)+svg_arrow(430,52,510,95)+svg_arrow(430,137,510,130)+svg_arrow(430,222,510,165)+svg_arrow(700,130,755,130),
        "This diagram is a proposed evaluation architecture, not executed LLM evidence. Full-test-split, call-matched comparisons and a separately calibrated development set are required. A meaningful phase variable must add predictive value beyond ordinary disagreement before a PLL analogy is claimed. Paid calls require the user's cost approval.")
    return "".join(out)


def overview(metadata, tables):
    """Derive the opening scientific scope and findings from executed grid rows."""
    grid = tables["combined_summary"]
    baseline = min(grid, key=lambda r: abs(r["alpha"]-.15)+abs(r["delta"]-.5))
    near = min(grid, key=lambda r: abs(r["alpha"]-.4)+abs(r["delta"]-.99))
    far = min(grid, key=lambda r: abs(r["alpha"]-.4)+abs(r["delta"]-.5))
    n = sum(int(r["n"]) for r in grid)
    content = '''<div class="lede">A phase-locked loop has a simple task: keep an oscillator aligned with an incoming signal. The difficult part is learning when noise produces harmless fluctuation, when a loop escapes to the next phase well, and when frequency mismatch removes the locked equilibrium altogether.</div>
    <p>This project follows that distinction from the textbook equation through the numerical update, independent-path experiments, uncertainty estimates and reproducible artifacts. The supplied pages of Kloeden and Platen establish the mathematical starting point. The large local simulation matrix tests the model's behavior; the synthetic communication study is a separate extension.</p>
    <h3>Questions that the experiments can answer</h3>
    <ol><li>How do phase variability, finite-horizon capture and completed well transitions change as noise intensity and normalized detuning increase?</li>
    <li>How accurately does the explicit stochastic integrator resolve this additive-noise system when it is compared on the same Brownian paths?</li>
    <li>How much do event counts depend on a defensible persistence rule, and what do selected longer runs reveal about rare events?</li>
    <li>Which synchronization effects survive changes in network topology, noise sharing, dropout and delay in a separately specified synthetic model?</li></ol>'''
    content += table(["Executed evidence", "Scope"], [
        ["PLL grid", f"{len(grid)} conditions; {n:,} complete independent paths within conditions"],
        ["Acquisition from original initial phase", f"{number(metadata.get('original_initial_trajectories'))} paths, separately recorded"],
        ["Selected long-horizon conditions", f"{number(metadata.get('long_conditions'))} conditions; {number(metadata.get('long_trajectories'))} paths"],
        ["Parameter sensitivity", f"{number(metadata.get('sensitivity_conditions'))} conditions; {number(metadata.get('sensitivity_trajectories'))} paths"],
        ["Coupled convergence", f"{number(metadata.get('convergence_trajectories'))} reference-path realizations"],
        ["Synthetic oscillators", f"{number(metadata.get('synthetic_conditions'))} conditions; {number(metadata.get('synthetic_trajectories'))} condition-runs with matched inputs"],
        ["Paid LLM inference", "Not executed; proposal and approval gate only"],
    ], "Execution scope. Reused E2/E3 slices of the grid are not extra independent evidence.")
    content += '<h3>Measured anchors for the rest of the report</h3>'
    content += paragraph(f'At the baseline grid cell (δ=0.5, α=0.15), terminal phase variance is <strong>{number(baseline.get("phi_variance"))} rad²</strong>, the finite-horizon capture fraction is <strong>{number(baseline.get("locking_probability"))}</strong>, and the completed-slip rate is <strong>{number(baseline.get("slip_rate"))} transitions per model time unit</strong>. These figures describe paths initialized at the stable phase; the original displaced initial state is examined separately.')
    content += paragraph(f'At α=0.4, the near-threshold cell δ=0.99 has terminal phase variance <strong>{number(near.get("phi_variance"))} rad²</strong> and slip rate <strong>{number(near.get("slip_rate"))}</strong>. At δ=0.5 with the same noise these are <strong>{number(far.get("phi_variance"))} rad²</strong> and <strong>{number(far.get("slip_rate"))}</strong>. The comparison is descriptive and condition-specific; all intervals and the complete matrix appear in Section F.')
    content += '''<h3>What the project contributes</h3><p>The useful contribution is a traceable computational study: a source-checked reduced model, an explicit test of the numerical implementation, a full noise–detuning matrix, operationally defined completed transitions, independent-trajectory uncertainty and a report that exposes the code and evidence. It does not claim to invent PLL theory, prove a new rare-event law, or establish that language agents are physical oscillators.</p>
    <p class="note">Read the evidence in order: the mathematical sections explain what each observable means; the methodology fixes what was measured; the results show what happened under that definition. Finite simulation time, numerical discretization and operational capture remain limits even when all planned local conditions finish.</p>'''
    return content


def foundations(root):
    """Explain the model, stability and stochastic discretization without hidden notation."""
    content = '''<p>The supplied scans identify the communication-system example in Chapter 7, Section 10. The second-order PLL equation is equation (10.4) on printed page 274. Pages 273 and 275 provide neighboring context. The page scans are embedded below so a reader can inspect the exact source instead of relying on an invented textbook citation.</p>
    <p>The equation is a reduced stochastic feedback model. Its time variable is measured in model units; no circuit-level calibration was supplied. The white-noise idealization replaces fast unresolved disturbances by a Gaussian process with independent increments. This is an assumption about the forcing, not a claim that every physical receiver has Gaussian or memoryless noise.</p>'''
    content += paragraph('The textbook writes detuning as ω−ω₀ and calls α a noise intensity. We define Δ=ω−ω₀ and explicitly choose <strong>unit-standard-Wiener forcing</strong>, Var[W(t+h)−W(t)]=h. The supplied PLL paragraph does not independently fix a white-noise covariance normalization. In this project α is an amplitude coefficient: its square controls covariance. A convention using covariance 2Dδ(t−s) must convert the product of the amplitude and √(2D), rather than silently retaining the same numerical α.')
    content += equation("balance", r"a\ddot\phi+\dot\phi+ab\sin\phi=\Delta-\alpha\xi(t),\qquad \xi(t)\,dt=dW(t)",
        "B1. The supplied second-order balance", [
            ["t", "Time in model units; dt is a short elapsed time."],
            ["φ(t)", "Unwrapped phase difference in radians; φ and φ+2π have the same circular angle but different accumulated cycles."],
            ["φ̇, φ̈", "Phase velocity in rad/time and phase acceleration in rad/time²."],
        ["a > 0", "Filter time constant a=RC (resistance × capacitance); controls damping and noise gain after division."],
            ["b > 0", "Feedback coefficient related to the signal-amplitude product; ab is the restoring-force amplitude in the balance."],
            ["Δ=ω−ω₀", "Input frequency ω minus reference frequency ω₀; Δ/(ab) controls equilibrium existence."],
            ["α ≥ 0", "Noise amplitude in the original balance; effective velocity diffusion is α/a."],
            ["ξ(t)", "Formal Gaussian white noise, interpreted only under integration."],
            ["W(t)", "Standard real Brownian motion, W(0)=0, Var[W(t)−W(s)]=t−s for t>s."],
        ], "The sine term pulls phase toward a periodic alignment. Velocity is damped. Detuning tilts that balance, while noise perturbs the acceleration. The minus sign is retained to reproduce the supplied convention. Replacing W by −W gives the same distribution, but retaining the sign matters for exact path replay.",
        '<code>PLLParameters</code> stores a, b, detuning and alpha. <code>drift</code> implements the deterministic balance and <code>diffusion</code> returns the signed velocity diffusion. No finite-valued array is interpreted as ξ(t) itself.')
    content += '<h3>Derive the state system one step at a time</h3><ol><li>Set v=φ̇ to convert the second-order equation to two first-order coordinates.</li><li>Rearrange a v̇=Δ−v−ab sin φ−α ξ(t).</li><li>Divide by a, multiply by dt and interpret ξ(t)dt as dW(t).</li><li>Keep φ unwrapped when tracking well transitions; wrap it only for circular errors.</li></ol>'
    content += equation("ito", r"d\phi=v\,dt,\qquad dv=\frac{\Delta-v-ab\sin\phi}{a}\,dt-\frac{\alpha}{a}\,dW(t)",
        "B2. The Itô stochastic differential equation", [
            ["X=(φ,v)ᵀ", "Two-dimensional state: phase and phase velocity."],
            ["dφ, dv", "Infinitesimal phase and velocity increments in the integral SDE sense."],
            ["fφ=v", "Phase drift, in radians per model time unit."],
            ["fv=(Δ−v−ab sin φ)/a", "Deterministic acceleration, combining detuning, damping and restoring feedback."],
            ["g=(0,−α/a)ᵀ", "One-column diffusion vector; one scalar Brownian driver enters velocity only."],
        ], "Noise kicks velocity. Phase accumulates that noisy velocity over time, so phase can be smoother than the Brownian forcing while still being random. The Itô and Stratonovich drift corrections agree here because the diffusion vector is constant.",
        '<code>pll_model.drift(state,p)</code> returns f; <code>diffusion(p)</code> returns g. <code>euler_maruyama</code> and the compiled kernel apply the same explicit update. The arrays have two state columns, never two independent Brownian channels.')
    content += equation("brownian", r"\Delta W_n=W(t_{n+1})-W(t_n)=\sqrt{h}\,Z_n,\quad Z_n\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\quad \operatorname{Var}(\Delta W_n)=h",
        "B3. Why the square root of the time step is necessary", [
            ["h=tₙ₊₁−tₙ > 0", "Numerical time step in model time units."],
            ["n", "Time-step index along one path."],
            ["ΔWₙ", "Brownian increment over that one time interval."],
            ["Zₙ", "Independent standard-normal variate; dimensionless."],
            ["iid", "Independent and identically distributed across disjoint increments for one Brownian path."],
        ], "Adding hZ instead would give variance h² per step, so the total noise would disappear as the mesh is refined. Adding Z without √h would make the accumulated noise diverge as the number of steps increases. With √h, the total increment variance over a fixed duration stays equal to that duration.",
        '<code>brownian_increments(n,dt,seed)</code> draws a normal array with standard deviation <code>np.sqrt(dt)</code>. Separate SeedSequence identities create independent trajectory streams. For convergence, coarse increments are sums of the existing fine increments, not fresh random draws.')
    content += equation("em", r"\phi_{n+1}=\phi_n+h v_n,\qquad v_{n+1}=v_n+\frac{h}{a}(\Delta-v_n-ab\sin\phi_n)-\frac{\alpha}{a}\Delta W_n",
        "B4. The implemented Euler–Maruyama update", [
            ["φₙ, vₙ", "Both components of the old numerical state at tₙ."],
            ["φₙ₊₁, vₙ₊₁", "Both components after one explicit stochastic step."],
            ["h", "The numerical step; h=0.001 for the full PLL matrix."],
            ["ΔWₙ", "A single Brownian increment shared by this velocity update."],
        ], "Advance position using the current velocity; advance velocity using the current phase and velocity plus a random kick. Both right-hand sides use old state values. Stability and accuracy therefore remain step-size questions, even though the random increment is generated exactly in law.",
        '<code>euler_maruyama</code> reads <code>phi,v=x[i]</code> before assigning either next-state component. The C++ kernel saves <code>oldphi</code> before updating phase, so its sine force also uses the old phase. The test suite compares Python and compiled results on identical increments.')
    content += equation("milstein", r"X_{n+1}=X_n+f(X_n)h+g(X_n)\Delta W_n+\tfrac12 Dg(X_n)g(X_n)\big((\Delta W_n)^2-h\big),\qquad Dg=0",
        "B5. Why Milstein coincides with Euler–Maruyama here", [
            ["Xₙ", "Two-coordinate numerical state."],
            ["f", "Drift vector defined in B2."],
            ["g", "Constant diffusion vector defined in B2."],
            ["Dg", "Jacobian of g with respect to state; every derivative is zero."],
            ["½ Dg g ((ΔWₙ)²−h)", "The scalar-noise Milstein correction, identically zero in this model."],
        ], "The correction accounts for changing diffusion strength as the state moves. Here noise gain never changes with phase or velocity, so there is no such correction. Calling the same update Milstein does not improve it.",
        '<code>milstein</code> intentionally delegates to <code>euler_maruyama</code>. A pathwise equality test is meaningful; a claim of two competing methods with different performance would be misleading. Additive noise can permit stronger convergence than the generic order-½ theorem, but a measured slope still needs a resolved reference and error regime.')
    content += '<h3>Equilibria and their stability</h3><p>Set α=0 and require both state derivatives to vanish. The phase equation gives v=0. Substitution into the acceleration equation gives sin φ=Δ/(ab). The strict inequality below identifies distinct stable and unstable branches. At equality the branches coalesce; treating that nonhyperbolic point as an ordinary stable locked state would be incorrect.</p>'
    content += equation("equilibria", r"\delta=\frac{\Delta}{ab},\quad |\delta|<1:\quad \phi_s=\arcsin\delta+2k\pi,\qquad \phi_u=\pi-\arcsin\delta+2k\pi,\qquad v_s=v_u=0",
        "B6. Stable and unstable phase branches", [
            ["δ", "Dimensionless normalized detuning."],
            ["k ∈ ℤ", "Integer well or cycle index; adjacent stable phases are separated by 2π rad."],
            ["φₛ", "Stable-phase branch with positive cosine."],
            ["φᵤ", "Unstable-phase branch with negative cosine."],
            ["arcsin", "Principal inverse sine, with values in [−π/2,π/2]."],
        ], "Below threshold the sine feedback can balance the constant detuning. Near |δ|=1, the stable phase approaches the unstable barrier and the restoring slope weakens. Above threshold no static balance exists. Existence of a stable local equilibrium does not by itself prove capture from every initial state in this second-order system.",
        '<code>equilibria(p)</code> returns a stable/unstable pair only for <code>abs(p.delta)&lt;1</code>. It returns <code>None</code> otherwise. <code>initial_phase</code> starts at the stable phase for locked-regime grid cells and at zero when no equilibrium exists.')
    content += equation("jacobian", r"J(\phi_*)=\begin{pmatrix}0&1\\-b\cos\phi_*&-1/a\end{pmatrix},\qquad \lambda^2+\frac{1}{a}\lambda+b\cos\phi_*=0",
        "B7. Linear stability is a two-dimensional calculation", [
            ["φ*", "An equilibrium phase on either branch, with v*=0."],
            ["J", "Jacobian of the deterministic drift with respect to (φ,v)."],
            ["λ", "A linearized growth/decay exponent, in inverse time units."],
            ["tr J=−1/a", "Negative damping trace for a>0."],
            ["det J=b cos φ*", "Positive on the stable branch, negative at the saddle branch."],
        ], "A negative trace and positive determinant give decay near the stable branch, either oscillatory or nonoscillatory. At the unstable branch the determinant is negative, so one direction grows and one decays: the equilibrium is a saddle. The saddle's stable manifold, not a vertical phase threshold, forms the local basin boundary.",
        '<code>eigenvalues(phi,p)</code> forms this matrix. <code>saddle_manifold</code> finds the negative-eigenvalue direction and integrates backward from a small displacement on each side. Backward integration is numerically delicate and the plotted branch is truncated; the detector does not treat it as an exact separatrix.')
    content += equation("potential", r"U(\phi)=-\Delta\phi-ab\cos\phi,\qquad E(\phi,v)=\tfrac12 av^2+U(\phi),\qquad \frac{dE}{dt}=-v^2\ \ (\alpha=0)",
        "B8. Tilted potential and deterministic energy loss", [
            ["U(φ)", "Potential-like landscape; U′=−Δ+ab sin φ."],
            ["E(φ,v)", "Mechanical energy-like quantity in the reduced model."],
            ["½av²", "Kinetic-like contribution from phase velocity."],
            ["dE/dt=−v²", "Deterministic damping removes energy along any smooth zero-noise solution."],
        ], "Substitute a v̇=−v−U′ and φ̇=v into the chain rule: a v v̇+U′v=−v². Because detuning tilts U, the landscape is not periodic even though its force is periodic. Noise can add and remove energy, so the zero-noise identity is not a stochastic pathwise conservation law.",
        '<code>potential</code> returns U for phase-portrait illustrations. A stochastic energy calculation would require Itô’s formula and an additional quadratic-variation contribution; the report does not reuse the deterministic energy derivative as an SDE identity.')
    content += equation("barrier", r"\Delta U=2ab\left[\sqrt{1-\delta^2}-|\delta|\arccos|\delta|\right],\qquad |\delta|<1",
        "B9. The downhill barrier shrinks near threshold", [
            ["ΔU", "Potential difference between a stable well and the adjacent downhill saddle."],
            ["|δ|", "Absolute normalized detuning; the downhill direction reverses with its sign."],
            ["arccos", "Principal inverse cosine."],
            ["a,b", "Positive model parameters already defined in B1."],
        ], "Evaluate U at the stable and downhill unstable phases, subtract, and simplify using cos(arcsin δ)=√(1−δ²) and π/2−arcsin|δ|=arccos|δ|. At zero detuning the barrier is 2ab; at the threshold it tends to zero. The barrier motivates an escape hypothesis but is not an automatically valid waiting-time formula for this underdamped noisy system.",
        '<code>barrier_height</code> computes this expression and returns zero when no locked well exists. The experiments estimate transitions directly. They do not fit an unjustified Kramers law to a few observed events.')
    content += '''<h3>Four quantities that should not be conflated</h3>
    <p><strong>Equilibrium existence</strong> is the algebraic condition |δ|&lt;1. <strong>Terminal capture</strong> is an operational finite-horizon event: the final wrapped phase is inside a π/4 core and the last-fifth mean velocity has magnitude below 0.1. <strong>Retained lock</strong> additionally requires no committed slip during the run. <strong>Rotation</strong> measures unwrapped cycles per unit time and remains meaningful when no stable wells exist. A path can end captured after slipping, so capture and retained-lock probabilities need not agree.</p>
    <p>A completed slip requires an adjacent-well core visit and continuous residence. It is a reproducible operational proxy for a full transition. The two-dimensional saddle manifold provides geometric context; the core/dwell rule deliberately avoids claiming that every phase threshold crossing is a transition.</p>'''
    content += '<details class="source-pages"><summary>Inspect the supplied textbook pages and exact provenance</summary>'
    for page in [273,274,275]:
        path = root / f"notes/textbook_p{page}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing provided textbook source page {page}")
        content += f'<figure><img loading="lazy" class="source-page" src="{data_uri(path)}" alt="User-supplied scan of Kloeden and Platen printed page {page}"/><figcaption>Supplied source page {page}. SHA-256: <code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code>. The scan is retained as provided; equation (10.4) is on page 274.</figcaption></figure>'
    content += '</details>'
    return content


MODULE_GUIDE = {
    "pll_model.py": ("The mathematical contract", "Validated a,b,Δ,α and one phase–velocity state.", "Drift, constant diffusion, equilibria, eigenvalues and the potential barrier.", "B1–B9", "This file decides what system is simulated. Solver and diagnostics consume the same parameter object so the restoring force and equilibrium threshold stay consistent.", "Do not replace ab by b in the original balance or assign an arcsine equilibrium outside |δ|<1."),
    "sde_solver.py": ("Path generation and deterministic reference", "Parameters, one Brownian-increment array, dt, initial state and sampling stride.", "Full or sampled numerical states, trajectory summaries and event times; a high-accuracy zero-noise reference.", "B2–B5", "It connects the model to both the readable Python integrator and the fast compiled kernel. Brownian increments are inputs, so paired methods can share the same forcing.", "Generating a new Brownian path for each time step comparison would destroy the pairing. Using a newly updated coordinate in the other drift changes the method."),
    "phase_slips.py": ("Completed-transition detection and basin context", "An unwrapped phase path, dt, stable phase and core/dwell settings; or model parameters for a saddle.", "Commitment times/directions and two approximate stable-manifold branches.", "B6–B7; E6", "The detector requires adjacent-well arrival and persistence. The manifold provides geometric context rather than an unvalidated numerical event boundary.", "Do not count each noisy boundary recrossing as a cycle slip. Do not run a locked-well detector where |δ|≥1."),
    "metrics.py": ("Independent-path uncertainty and censoring", "Per-trajectory scalar arrays, binary counts, event/censor times and confidence level.", "Mean, Wilson or bootstrap intervals and Kaplan–Meier first-event summaries.", "E1–E5", "These routines treat complete paths as independent sampling units. A time sample from a path is not another path.", "An all-zero event bootstrap is degenerate. A waiting-time average over observed slips ignores censored paths."),
    "statistical_analysis.py": ("Saved-output analysis", "Raw trajectory-summary rows and complete condition metadata.", "Condition tables, intervals, survival summaries and paired convergence diagnostics.", "E1–E5", "It links raw summaries to the tables displayed in the report without rerunning or silently changing the simulation protocol.", "Do not pool reused grid slices as independent evidence or drop censored paths from waiting-time analysis."),
    "experiment_runner.py": ("Reproducible execution and checkpoint orchestration", "Preregistered configuration, master seed, source identity, condition IDs and output root.", "Raw independent-path rows, representative trajectories, condition metadata and completion records.", "B3–B4; Section E", "It creates reproducible random-stream identities, calls the solver, preserves raw outcomes and coordinates complete ensembles. Its recorded metadata determines the report's execution claims.", "An existing filename is not a valid checkpoint until configuration and source identity agree. Reusing a condition does not create a second independent experiment."),
    "visualization.py": ("Figures from measured outputs", "Saved CSV/JSON/NPZ analysis and representative states.", "Labeled scientific SVG/PNG figures and captions.", "B6–B9; Section F", "It transforms completed data into visual evidence. Conceptual teaching SVGs belong to the report generator and are labeled separately.", "A confidence interval for an ensemble mean is not a 95% containment band for individual paths. Missing values in unlocked cells must stay unavailable."),
    "analysis_supplement.py": ("Descriptive distribution summaries and selected contrasts", "Saved independent-path CSV rows and existing condition summaries.", "Enriched medians/SDs and independent-cohort bootstrap effect-size tables.", "E1–E3; F4", "It rereads raw rows to add distribution summaries and compute right-minus-left contrasts with separate resampling of the independent cohorts.", "Selected descriptive contrasts do not become preregistered significance tests. A pooled-SD standardized difference is unavailable when both cohorts are constant."),
    "deterministic_validation.py": ("Independent zero-noise mesh validation", "The deterministic model, displaced initial states, candidate meshes and DOP853 reference.", "Maximum and endpoint state errors over T=50 at two detunings.", "B4; E5", "It isolates deterministic discretization from stochastic variation and records full-path maximum differences, rather than relying on a near-equilibrium terminal match.", "Tiny endpoint error after damping can conceal an earlier larger transient error; inspect the maximum over matched sample times."),
    "latex_report.py": ("The concise formal course report", "Completed condition summaries, synthetic results, verified bibliography and saved charts.", "Seven-section LaTeX source using actual numbers and references.", "The same equations and evidence as this longer teaching report", "It produces the teacher's formal structure as a separate readable artifact, rather than treating the HTML as a replacement for the required LaTeX deliverable.", "Never copy stale numbers into a final manuscript; regenerate from the same saved evidence and compile the document."),
    "verify_outputs.py": ("Cross-file output integrity", "Completed configurations, raw rows, events, summaries and manifests.", "A verification report or an explicit failure at the inconsistent artifact.", "Reproducibility rather than a new dynamical equation", "It checks whether files, counts, finite outcomes and expected experimental design agree. Integrity supports provenance; it does not validate a physical model or prove statistical accuracy.", "A file's presence is weaker than checking its content, row count, source identity and saved digest."),
    "reproduce.py": ("Re-execution and final evidence gates", "Existing project or a full-run request and the current interpreter.", "Validated full execution, final test logs, complete metadata, merged captions and regenerated HTML/LaTeX/PDF.", "Entire workflow", "It runs commands with failure propagation, finalizes metadata only after the synthetic matrix is complete, and regenerates reports from verified outputs.", "The reports-only mode still checks outputs and tests; it must not invent a completion flag for unfinished experiments."),
    "synthetic_agents.py": ("The separate phase-network extension", "Network topology, frequencies, coupling, noise correlation, dropout/delay settings and seeds.", "Coherence, disagreement and trajectory-level network summaries.", "E6–E7", "It studies explicit oscillator phases on a communication graph. It is related through synchronization ideas, but is not a derivation of language-agent behavior from the PLL equation.", "Shared noise can alter observed coherence independently of communication. Changing node count or normalization changes how topology comparisons should be interpreted."),
    "report_generator.py": ("A fail-closed offline evidence document", "Completed metadata, summary tables, measured series, saved figures, notes and source files.", "One standalone HTML file with embedded images, MathJax, code, data and controls; an asset manifest.", "All equations are explained alongside their code relationships", "It never performs a simulation. It verifies that required evidence exists before writing a report, then composes mathematical explanations and actual results.", "Never insert invented values for a missing result. The explanatory SVGs are conceptual; figures in Section F must trace to executed files."),
    "__init__.py": ("Package boundary", "Python import requests.", "The src package namespace.", "No numerical equation", "It lets scripts use relative imports consistently.", "Running a package module as a loose script can break relative imports; use the documented module command."),
}

FUNCTION_GUIDE = {
    "__post_init__": ("Reject nonfinite parameters and require a,b>0 and α≥0 before any dynamics are evaluated.", "The frozen parameter object is valid only after every stated inequality passes.", "Permitting a=0 would divide by zero; allowing negative α obscures the agreed noise convention."),
    "delta": ("Normalize detuning by the maximum restoring-force amplitude ab.", "The dimensionless ratio determines the equilibrium-existence threshold.", "Δ and δ are different quantities. The compiled kernel's internal variable name delta carries physical Δ."),
    "drift": ("Evaluate [v,(Δ−v−ab sinφ)/a] at one old phase–velocity state.", "The first coordinate integrates velocity; the second combines drive, damping and sine feedback.", "The phase coordinate is unwrapped; applying sine is periodic but storing only a wrapped phase loses cycle count."),
    "diffusion": ("Return the constant signed diffusion vector [0,−α/a].", "Only velocity receives a direct Brownian kick; Dg is zero.", "Adding a second independent phase noise would define a different SDE."),
    "equilibria": ("Return the stable phase and adjacent unstable phase only below |δ|=1.", "The strict domain test occurs before arcsin so unlocked states have no invented equilibrium.", "At equality the branch is nonhyperbolic; it is not an ordinary stable locked phase."),
    "eigenvalues": ("Form the deterministic 2×2 Jacobian and calculate local growth exponents.", "Positive cosine gives negative-real-part exponents; negative cosine gives a saddle.", "A stable local linearization does not establish global acquisition from every initial state."),
    "potential": ("Evaluate −Δφ−ab cosφ for scalar or array phase input.", "Its derivative supplies the restoring and detuning forces in the damped mechanical analogy.", "The potential is tilted when Δ≠0; it is not a periodic scalar despite periodic sine force."),
    "barrier_height": ("Compute the downhill stable-to-saddle potential barrier for |δ|<1.", "The expression approaches zero at the threshold and motivates increased escape susceptibility.", "A barrier value is not a validated escape-rate law for this inertial stochastic model."),
    "initial_phase": ("Start at the stable phase where it exists, otherwise at phase zero.", "This controls the distinction between equilibrium-fluctuation grid experiments and displaced-state acquisition runs.", "Starting all near-threshold cells at the same absolute phase would mix acquisition transients with a different initial condition."),
    "brownian_increments": ("Validate step count and dt, create a seeded generator and draw n independent N(0,dt) increments.", "The square-root scaling preserves Brownian variance under mesh refinement.", "Do not multiply by dt a second time when applying an increment that already has variance dt."),
    "coarsen_increments": ("Reshape a fine Brownian path into contiguous blocks and sum each block.", "Each coarse increment covers exactly the same time interval as its fine constituents.", "A coarsening factor must divide the path length. Averaging increments would use the wrong Brownian variance."),
    "euler_maruyama": ("Allocate a two-column path and update phase/velocity explicitly from the old state for every increment.", "The returned array includes the initial state, so it has len(dw)+1 rows.", "Do not evaluate the sine force at the just-updated phase or substitute fresh noise when comparing methods."),
    "milstein": ("Use the Euler–Maruyama path because all diffusion derivatives vanish.", "The equality is a mathematical property of this model, rather than an omitted acceleration method.", "Never claim a Milstein improvement from this identical update."),
    "deterministic": ("Integrate the drift with DOP853 at requested sample times and strict tolerances.", "This supplies an independent zero-noise numerical reference, with a solver success check.", "It is a high-accuracy numerical reference, not a symbolic exact solution or stochastic integrator."),
    "load_kernel": ("Compile the C++ source when necessary, load its shared library and declare exact ctypes array/scalar signatures.", "Contiguous float64 and int32 buffers make the Python/C++ memory contract explicit.", "Stale binaries or incorrect ctypes signatures can silently invalidate a fast run; source identity and Python parity tests matter."),
    "compiled_path": ("Validate step/stride shape, allocate state/summary/event buffers and call the compiled path kernel.", "It returns copied valid event slices and rejects nonfinite sampled states.", "Downsampled stored paths cannot replace full-step event detection. The kernel detects events at every integration step."),
    "detect_slips": ("Track occupied and candidate adjacent wells, reset dwell age when the core is left, and commit after continuous residence.", "The event direction is ±1 and its time is the end of the required dwell interval.", "Repeated threshold crossings without persistent arrival are not independent completed transitions."),
    "saddle_manifold": ("Find the saddle's negative-eigenvalue direction, perturb both signs and integrate backward to trace a local stable manifold.", "The branches are numerical geometric context and are truncated to a bounded plotting region.", "Backward integration can amplify error. This routine is not used as an exact or automatically validated separatrix detector."),
    "mean_ci": ("Calculate the independent-path sample mean and a Student-t standard-error interval.", "For fewer than two paths, Monte Carlo uncertainty is unavailable and bounds are NaN.", "A pointwise confidence interval for the mean is not the spread of individual trajectories."),
    "wilson_interval": ("Invert the normal-score relation for a binary independent-path outcome.", "The center and half-width remain meaningful at zero or all successes and are clipped to [0,1].", "The denominator is trajectories, not time samples. An interval is still needed when the observed fraction is zero or one."),
    "bootstrap_ci": ("Resample complete path values with replacement, calculate the statistic in bounded batches and take percentile bounds.", "The resampling seed makes the reported interval reproducible without altering simulation paths.", "Identical or all-zero inputs give a degenerate interval; that does not establish absence of rare events."),
    "kaplan_meier": ("At each observed event time, update survival by the event fraction among paths still at risk, then integrate survival to the horizon.", "It returns a right-censor-aware restricted mean and a median only if survival reaches one half.", "A conditional mean over the paths that slipped is not an unconditional mean waiting time."),
    "source_hash": ("Hash the exact numerical-model, solver, kernel and inference sources used by the PLL runner.", "Checkpoint fingerprints combine this identity with the condition configuration.", "A changed numerical source invalidates an earlier checkpoint even when the condition label stays the same."),
    "design": ("Construct every preregistered noise–detuning, long-horizon, parameter-sensitivity and original-state condition.", "It assigns 500 or selected 1000 stochastic trajectories, a deterministic single path at α=0, and unique condition IDs.", "E2/E3 slices reuse grid data and cannot be counted as independent reruns."),
    "run_condition": ("Validate the checkpoint identity or simulate every path in one condition, then preserve per-path rows, events, sampled ensemble statistics and configuration.", "Each path's seed derives from master/condition/trajectory identity. All path summaries are analyzed only after complete execution.", "Sampled representative curves are not the complete raw event process; event detection runs at every step."),
    "export_group": ("Read completed condition summaries and export one consistent group CSV.", "It omits nested survival arrays from the flat table while preserving them in condition JSON.", "Exporting a subset from missing or unfinished cells would not meet the full-matrix protocol."),
    "original_replay": ("Import the preserved baseline code and compare it with the compiled path for exact original seeds, saving the differences and seed-42 path.", "The old random seed and displaced initial phase are held identical to isolate implementation agreement.", "A reproduction that silently starts at the stable phase would change the original experiment."),
    "convergence_case": ("Create 1000 or 500 fine Brownian paths, aggregate each to five coarser meshes, and compare endpoint/path errors with the numerical reference.", "Bootstrap and paired t intervals preserve within-path pairing. The same reference is used for all five candidate steps.", "The fitted rate is a finite-range empirical slope against a finite numerical reference, not a proof of the asymptotic order."),
    "summarize": ("Transform independent trajectory rows into condition means, terminal variance, capture/retained-lock probabilities, completed-slip rates and censored first-event summaries.", "Nonexistent locked-well outcomes are marked unavailable above threshold. Zero-event Poisson exposure bounds are labeled approximations.", "Do not conflate across-path terminal variance with the tail within-path variance, or finite-horizon capture with guaranteed permanent lock."),
    "topology_matrix": ("Construct the actual network adjacency for complete, ring, star or connected sparse communication.", "Degree normalization uses the original row degree, so dropout reduces received coupling rather than renormalizing it away.", "For three nodes the undirected ring and complete graph coincide, so those cells are duplicate mathematical topology evidence."),
    "order_parameter": ("Calculate the length of the average unit phase vector.", "R=1 means all circular phases coincide; low R indicates cancellation of directions.", "R alone can miss structured clusters and is not correctness or answer agreement for language agents."),
    "pairwise_disagreement": ("Measure average 1−cos phase difference over unordered node pairs.", "For equal weights it equals N/(N−1)(1−R²), so it is algebraically related to coherence.", "Do not present this quantity and R as independent information about the network state."),
    "simulate_networks": ("Validate network configuration, generate matched initial/noise/dropout streams, call the compiled oscillator kernel and return one metric row per network trajectory.", "The kernel evolves the global phase and centered relative phases separately. Pure common additive noise cancels from relative dynamics exactly.", "The delayed term uses the stale relative error for both nodes. Sender-only delay would define a different model and could break common-noise cancellation."),
    "summarize_networks": ("Calculate mean, standard deviation, median and Student-t confidence intervals across complete independent network-trajectory aggregates.", "Time samples are averaged within each network before cross-network inference.", "Network cells deliberately share matched inputs; counts across all cells are not that many independent network realizations."),
    "full_matrix": ("Enumerate three network sizes, four topology labels, three noise correlations, two dropout levels and two delays.", "The Cartesian design contains 144 cells with 500 networks per cell.", "N=3 ring and complete are the same graph; the 72,000 runs are paired across configurations using 1,500 unique input identities."),
    "_stream_seeds": ("Derive three reproducible per-network stream seeds from master seed, node count, replicate and stream identity.", "Cell labels are deliberately absent so topology and impairment comparisons use matched inputs.", "Shared seeds across cells establish pairing; they do not create independent replications across those cells."),
    "_paired_effects": ("Subtract each no-impairment network aggregate from its matched dropout/delay aggregate and form a confidence interval for that paired difference.", "This controls initial phases and random-stream identity within each comparison.", "Separate marginal confidence intervals do not replace a paired difference interval. The many descriptive intervals are not multiplicity-adjusted significance tests."),
    "run_full": ("Run or checksum-validate every synthetic cell, preserve per-network rows/configuration/samples and complete the matrix metadata only after all artifacts are written.", "The manifest records source/configuration hashes, elapsed time and the exact paired design.", "An artifact's filename alone is not enough to validate a checkpoint; stored hashes and configuration must agree."),
    "enrich_tables": ("Read raw per-path rows for each completed family and add terminal/tail medians, sample SDs and phase-deviation spread to the existing summary table.", "The medians and SDs describe the distribution across paths, distinct from uncertainty in a mean.", "Do not interpret a sample SD as a confidence-interval half-width; they scale differently with trajectory count."),
    "effect_sizes": ("Bootstrap right-minus-left differences for four selected independent-cohort contrasts and three path metrics, then compute a pooled-SD standardized difference.", "Each cohort is resampled independently; the resulting interval describes a between-condition mean difference.", "These selected contrasts are descriptive. An all-equal binary cohort cannot supply a nonzero pooled SD."),
    "baseline": ("Load the original-initial-state ensemble and exact seed-42 replay, compare deterministic acquisition and plot actual phase/velocity means, variance and phase portrait.", "The chart distinguishes a representative path from an estimated mean and its pointwise uncertainty.", "A mean interval does not contain 95% of individual trajectories; the original displaced state differs from the stable-phase grid start."),
    "sensitivities": ("Read the full grid, extract fixed-detuning/noise slices and plot the complete interaction matrix with unavailable locked-well cells masked.", "This visualizes the same measured grid in several views, without creating extra evidence.", "Do not count a slice as a separately simulated ensemble or color an unavailable slip metric as a zero."),
    "slip_and_geometry": ("Plot actual long-horizon first-slip survival, paired detector counts, a measured slipping path and approximate deterministic saddle-manifold geometry.", "The plots connect operational events with phase-space context while retaining censoring.", "The deterministic local manifold is not an exact stochastic event detector; commitment time includes dwell."),
    "convergence_and_parameters": ("Plot coupled error/bias diagnostics and the a/b/noise sensitivity matrix from saved tables.", "It displays finite-reference refinement and joint mechanistic parameter changes.", "The fitted strong slope does not prove a weak rate or long-horizon event-count convergence."),
    "brownian": ("Generate a reproducible illustrative Brownian path and increment-scale chart using the recorded example seed.", "It teaches Var(ΔW)=dt separately from the large experimental cohorts.", "One illustration cannot establish the variance law empirically; the independent-increment preflight supplies the numerical check."),
    "condition": ("Read one condition's saved configuration and ensemble arrays for scientific plotting.", "This boundary keeps visualization connected to its actual execution identity.", "Selecting a convenient curve without its configuration would detach the figure from its cohort and initial state."),
    "save": ("Write the current Matplotlib scientific figure into the saved results-figure directory.", "The SVG is later embedded byte-for-byte in this standalone report.", "A saved chart needs a caption, axis units and the corresponding data provenance before it supports a scientific claim."),
    "finalize": ("Require the completed synthetic matrix, rerun the full test suite, merge actual figure captions and hash all PLL raw artifacts before final report generation.", "The completion metadata is an explicit post-execution gate rather than an assumption inferred from a filename.", "Tests protect defined behaviors; source and artifact hashes establish identity, not hardware realism or global capture."),
    "verify": ("Check completed output cohorts against expected counts, schema, seeds, finite state summaries and saved integrity evidence.", "This is a consistency audit over recorded scientific artifacts.", "A consistency pass does not turn an operational detector into an exact mathematical separatrix or validate a proposed LLM phase mapping."),
}


def statement_explanation(node):
    """Describe an actual AST statement and preserve its exact executable expression."""
    expression = ast.unparse(node)
    compact = expression.splitlines()[0]
    if len(compact) > 210:
        compact = compact[:207] + "..."
    if isinstance(node, (ast.For, ast.While)):
        return f"Repeat the following block over the stated index or condition: <code>{esc(compact)}</code>. Each iteration uses the current state of the variables shown in the source."
    if isinstance(node, ast.If):
        return f"Check the explicit branch condition <code>{esc(ast.unparse(node.test))}</code>; only the corresponding branch runs."
    if isinstance(node, ast.Return):
        return f"Return <code>{esc(ast.unparse(node.value) if node.value else 'None')}</code> to the caller. This is the actual output contract in the source."
    if isinstance(node, ast.Raise):
        return f"Stop with a clear exception if the stated validation or execution condition fails: <code>{esc(compact)}</code>."
    if isinstance(node, (ast.With, ast.Try)):
        return f"Enter the resource/error-handling block: <code>{esc(compact)}</code>. The complete scope and cleanup are visible in the source excerpt."
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"Define local helper <code>{esc(node.name)}</code>; its body runs only when called."
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return "Declare the function's documentation string. It describes the contract and does not change the numerical state."
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return f"Evaluate the right-hand expression and store/update the named variable(s): <code>{esc(compact)}</code>."
    return f"Execute <code>{esc(compact)}</code>. The ordering relative to the surrounding statements is part of the implementation."


def source_functions(tree):
    """Collect nested and class-qualified definitions in source order."""
    found = []
    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = prefix + child.name
                found.append((qualified, child))
                visit(child, qualified + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)
    visit(tree)
    return sorted(found, key=lambda pair: pair[1].lineno)


def code_walkthrough(root):
    """Explain each actual Python module/function and expose its complete current source."""
    paths = sorted((root / "src").glob("*.py"))
    content = '''<p>The walkthrough below is built from the actual files present at report-generation time. Signatures, statement order, return expressions, line numbers and full code are extracted with Python's abstract syntax tree. The authored mathematical commentary explains why those operations matter. Source hashes in Section J identify the precise files shown here.</p>
    <div class="math-code-map">'''
    content += table(["Mathematics / evidence", "Implementation", "Data crossing that boundary"], [
        ["SDE drift and constant diffusion (B2)", "pll_model → sde_solver", "Validated parameters and old (φ,v) state"],
        ["Brownian scaling / paired paths (B3)", "brownian_increments → coarsen_increments", "Fine increments; coarse increments are exact sums"],
        ["Explicit update (B4)", "euler_maruyama ↔ kernel.cpp", "Same forcing; parity tests establish agreement"],
        ["Completed transition (C09)", "phase_slips / kernel → raw per-path summary", "Core, dwell time, event direction and commitment time"],
        ["Independent-path inference (E)", "metrics / statistical_analysis → CSV", "Means, Wilson intervals, bootstrap rows and censored event times"],
        ["Executed evidence (F)", "visualization / report_generator", "Completed raw artifacts; no substitute placeholder curves"],
    ], "Mathematics-to-code dependency map") + '</div>'
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        guide = MODULE_GUIDE.get(path.name, (path.stem.replace("_", " ").title(), "Arguments listed in the exact signatures below.", "Return expressions and saved outputs shown below.", "See the function-level mathematical relationship.", ast.get_docstring(tree) or "The complete current source is included below.", "Check validation, path independence, and units at this module boundary."))
        functions = source_functions(tree)
        imported = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                imported.append(("."*node.level)+(node.module or "") + ": " + ", ".join(alias.name for alias in node.names))
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        content += f'<article class="module" id="module-{esc(path.stem)}"><h3><code>src/{esc(path.name)}</code><span>{esc(guide[0])}</span></h3>'
        content += paragraph(esc(guide[4]))
        content += table(["Contract", "Explanation"], [["Inputs", guide[1]], ["Outputs", guide[2]], ["Mathematical role", guide[3]], ["Interactions / imports", "; ".join(imported) or "No imports"], ["Common mistake", guide[5]]])
        if ast.get_docstring(tree):
            content += paragraph(f'<strong>Module documentation.</strong> {esc(ast.get_docstring(tree))}')
        for qualified, function in functions:
            doc = ast.get_docstring(function) or ""
            authored = FUNCTION_GUIDE.get(function.name)
            purpose = authored[0] if authored else (doc.split("\n\n")[0] if doc else f"Carry out the {function.name.replace('_',' ')} operation in {path.name}; the exact operations and return contract follow.")
            math_role = authored[1] if authored else guide[4]
            mistake = authored[2] if authored else guide[5]
            signature = f"{qualified}({ast.unparse(function.args)})"
            function_source = "\n".join(source.splitlines()[function.lineno-1:function.end_lineno])
            content += f'<details class="function"><summary><code>{esc(signature)}</code><span>lines {function.lineno}–{function.end_lineno}</span></summary>'
            content += paragraph(f'<strong>Purpose.</strong> {esc(purpose)}')
            content += paragraph(f'<strong>Inputs.</strong> The exact signature is <code>{esc(signature)}</code>. Default values shown here are read from the current source. Parameters passed through a model object keep a, b, detuning and alpha together; path arrays and summary rows are identified by their names and the enclosing module contract.')
            returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
            return_text = "; ".join(ast.unparse(node.value) if node.value else "None" for node in returns[:6]) or "No explicit return; effects are the assignments, writes or exceptions shown in the source."
            content += paragraph(f'<strong>Outputs.</strong> <code>{esc(return_text)}</code>.')
            content += paragraph(f'<strong>Mathematical / implementation role.</strong> {esc(math_role)}')
            if doc:
                content += f'<blockquote>{esc(doc)}</blockquote>'
            content += '<h5>Execution in source order</h5><ol class="execution">'
            for statement in function.body:
                content += f'<li><span class="line-ref">line {statement.lineno}</span> {statement_explanation(statement)}</li>'
            content += '</ol>'
            content += paragraph(f'<strong>Implementation choice and common mistake.</strong> {esc(mistake)}')
            content += highlight(function_source, PythonLexer(), HtmlFormatter(cssclass="code-highlight", linenos="table", linenostart=function.lineno))
            content += '</details>'
        content += '<details class="full-source"><summary>Read the complete file, with line numbers</summary>' + highlight(source, PythonLexer(), HtmlFormatter(cssclass="code-highlight", linenos="table")) + '</details></article>'
    content += numerical_loop(root)
    return content


def numerical_loop(root):
    """Explain the real compiled numerical loop line by line, including event tracking."""
    path = root / "src/kernel.cpp"
    if not path.exists():
        raise FileNotFoundError("The executed numerical kernel is required for its line-by-line walkthrough")
    source = path.read_text()
    meaning = [
        ("oldphi=phi", "Save the old phase. Both drift coordinates must use the same state at the beginning of this explicit step."),
        ("phi+=v*h", "Advance phase by old velocity × elapsed time. There is no direct phase Brownian kick in the supplied model."),
        ("v+=(", "Compute the deterministic acceleration from physical detuning Δ, old velocity, old sine force and division by a; then add the signed velocity Brownian kick −(α/a)dw[i]. Here the C++ argument called delta is physical Δ, not normalized δ."),
        ("double time=", "Convert the completed step index to model time. Commitment event times are recorded on this mesh."),
        ("%stride", "Store a sample only at the requested stride. This reduces disk/memory storage; every intervening integration and detector step still runs."),
        ("tailstartphi=phi", "Record phase at the start of the final 20% window so accumulated rotation has a matched time interval."),
        ("tailn++", "Accumulate last-fifth mean velocity, online Welford phase variance, and circular phase error without storing every time point."),
        ("if(wells)", "Only evaluate locked-well transitions if |Δ/(ab)|<1. Above threshold the event/lock outputs are not applicable."),
        ("int k=", "Assign the nearest stable-center index using unwrapped phase; periodic angle alone would lose the occupied-cycle identity."),
        ("r<3", "Evaluate primary, loose and strict detector rules on the same path. These are paired definitions, not independent trajectory ensembles."),
        ("k!=occupied", "A candidate must be in a different well and within that well's core. Crossing a barrier alone does not satisfy the rule."),
        ("int target=", "Require the candidate to be the adjacent well in the displacement direction, so one event is one completed ±2π transition."),
        ("target!=k", "Reject a nonadjacent jump under the stepwise detector. This also exposes why resolving the path and numerical-stability tests matter."),
        ("candidate[r]!=target", "Restart residence time whenever the candidate well changes. Separate visits cannot be added into a false continuous dwell."),
        ("age[r]+=h", "Increase continuous core residence by one time step, using model time rather than the number of stored samples."),
        ("age[r]+1e-12", "Commit only once the required dwell is reached. The small tolerance guards roundoff in repeated time additions."),
        ("int direction=", "Count the adjacent transition, update the occupied well and reset candidate/dwell state."),
        ("if(r==0)", "For the primary rule, preserve each event time/direction, the first event, and signed counts. Loose and strict rules remain sensitivity summaries."),
        ("else{candidate", "Leaving the candidate core resets the dwell. The detector does not count temporary arrival as completion."),
        ("summary[0]", "Write terminal phase/velocity and tail summary statistics. The tail variance is within-path temporal variance, distinct from across-path terminal phase variance."),
        ("summary[5]", "Write forward/backward transition counts or NaN if wells do not exist. The first-slip sentinel is interpreted as right censoring downstream."),
        ("summary[8]", "Write loose/strict detector counts for paired rule-sensitivity comparisons."),
        ("bool captured", "Define terminal capture from wrapped final phase error and small tail mean velocity; this is an operational finite-horizon criterion."),
        ("summary[10]", "Store capture and retained-lock indicators; retained lock additionally requires zero primary committed events."),
        ("summary[12]", "Convert unwrapped tail phase change to cycles per model time by dividing by window length and 2π."),
        ("*nevents=", "Return the valid event-buffer length through the pointer contract; Python copies only the filled part."),
    ]
    rows = []
    for line_number, line in enumerate(source.splitlines(),1):
        stripped = line.strip()
        if not stripped or stripped in {"}", "};"}:
            continue
        explanation = next((text for token,text in meaning if token in stripped), None)
        if explanation is None:
            if stripped.startswith("#include"):
                explanation = "Include standard numerical/type definitions required by the compiled library."
            elif 'extern "C"' in stripped:
                explanation = "Expose a stable C-callable entry point. Input arrays carry Brownian increments; output buffers carry sampled states, summaries and event records."
            elif "pi=" in stripped:
                explanation = "Define π and 2π once to keep phase-well indexing and circular operations consistent."
            elif "bool wells" in stripped:
                explanation = "Check the strict equilibrium-existence threshold using physical detuning divided by ab."
            elif "double stable" in stripped:
                explanation = "Use arcsin of normalized detuning only in the locked regime; otherwise zero is an initialization convention, not a stable phase."
            elif "radius[3]" in stripped:
                explanation = "Set primary/loose/strict cores to π/4, π/2, π/6 and residence times to 0.5, 0.25, 1.0."
            elif "occupied[r]=" in stripped:
                explanation = "Initialize each detector's occupied well from the actual unwrapped initial phase and stable center."
            elif "for(int i=0" in stripped:
                explanation = "Run exactly n explicit steps, consuming one Brownian increment per iteration."
            elif "path[0]=" in stripped:
                explanation = "Preserve the initial state before integration and initialize the stored-sample cursor."
            else:
                explanation = "Initialize or maintain the named state, event counters and online accumulators. The values are local to one independent path."
        rows.append([str(line_number), f'<code>{esc(stripped)}</code>', esc(explanation)])
    return ('<article class="module"><h3><code>src/kernel.cpp</code><span>The numerical loop, line by line</span></h3>'
            '<p>The compiled kernel changes execution speed, not the model. NumPy generates the Brownian arrays; C++ consumes them in the same order as the reference Python solver. The following annotation follows the exact current source. Variables called <code>delta</code> inside this low-level function carry Δ, while <code>PLLParameters.delta</code> is the normalized ratio δ.</p>'
            + table(["Line", "Actual source", "Why this line exists"], rows, "Exact numerical-loop annotation", raw=True)
            + '<details><summary>Read the complete compiled kernel</summary>' + highlight(source,CppLexer(),HtmlFormatter(cssclass="code-highlight",linenos="table")) + '</details></article>')


def methods(root, tables):
    """Document preregistered condition counts, estimands and independent-path inference."""
    content = '''<p>The full local design was recorded before its execution. Numerical tests were a prerequisite; they were not substituted for the experiment matrix. The PLL study uses a vectorized random-number interface and a compiled single-path loop, with bounded concurrent condition execution. The synthetic oscillator protocol is separate.</p>
    <h3>Parameter and control table</h3>'''
    content += table(["Quantity", "Values / rule", "Reason and control"], [
        ["a,b", "1,1 for E1–E6", "Fix damping and feedback while noise and detuning change."],
        ["Normalized detuning δ", "0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.01, 1.05", "Resolve stable wells, near-threshold weakening and the unlocked regime."],
        ["Noise amplitude α", "0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40", "Deterministic control through substantial stochastic disturbance."],
        ["Primary horizon / step", "T=50; h=0.001", "50,000 integration steps per path; detector evaluated at every step."],
        ["Grid trajectories", "500 per stochastic cell; 1000 for selected baseline/near-threshold cells", "63 stochastic cells plus 9 single-path deterministic controls; total 39,009."],
        ["1000-path selection", "δ=0.5 with α=0.10,0.15,0.40; δ=0.90,0.95,0.99 with α=0.10,0.15,0.30,0.40", "Predeclared important comparisons; no outcome-based cohort enlargement."],
        ["Initial state", "φ₀=arcsin δ,v₀=0 below threshold; φ₀=0,v₀=0 above", "Study fluctuations about the local well; do not invent a stable phase above threshold."],
        ["Original acquisition", "φ₀=arcsin(0.5)+0.5,v₀=0; n=1000", "Separate displaced-state reproduction and exact original seeds 0–4,42."],
        ["Long horizon", "δ=0.75,0.90,0.95 × α=0.15,0.30; n=1000,T=200", "Selected rare-event follow-up, not an exhaustive long-horizon grid."],
        ["a/b sensitivity", "a,b∈{0.5,1,2}, α∈{0.15,0.30}, Δ=0.5; n=500", "18 cells; changing a/b also changes normalized detuning, damping and effective noise."],
        ["Numerical convergence", "h=0.01,0.005,0.0025,0.00125,0.000625; reference 0.00015625; T=10", "1000 baseline and 500 nonlinear-stress paths; exact Brownian aggregation."],
        ["PLL random identity", "SeedSequence([20261113,condition_id,trajectory_id]) → recorded uint64 seed → PCG64", "Independent streams within cells; a saved trajectory seed reproduces the same forcing."],
    ], "Preregistered PLL design; all time values are in model time units")
    experiments = [
        ("E1. Baseline and original acquisition", "How do deterministic relaxation, noisy sample paths and an ensemble differ at a=b=1, Δ=0.5, α=0.15?", "The stable equilibrium attracts the zero-noise solution, while stochastic trajectories fluctuate and their mean is estimated with finite uncertainty.", "Use the 1000-path baseline cell; compare a representative stochastic path, mean/variance and phase portrait. Run another 1000 paths from the original phase displacement and replay six exact original seeds. Compare zero-noise integration with SciPy DOP853 independently of this stochastic ensemble.", "Terminal phase/velocity moments, pointwise mean intervals, capture/retained lock and exact replay errors."),
        ("E2. Noise sensitivity", "How does α alter fluctuations, operational capture and completed slips at δ=0.5?", "Increasing the amplitude broadens the state distribution; whether completed escapes become observable is an empirical question at T=50.", "Take the eight fixed-detuning cells from the complete grid; use one deterministic control and 500/1000 independent stochastic trajectories as predeclared. These are data slices, not new runs.", "Terminal phase variance, circular phase deviation, tail velocity, capture intervals, slips and censor-aware waiting outcomes."),
        ("E3. Detuning sensitivity", "How does behavior change as δ approaches and crosses one?", "The stable restoring slope and downhill barrier weaken as δ approaches one. Above one, static locking ceases and rotation becomes the relevant observable.", "Take the nine α=0.15 cells; retain all other controls. Start at the stable phase where it exists and zero otherwise. Never assign a locked-well phase slip to |δ|≥1.", "Variance, tail velocity, capture, completed transitions and mean rotation, with unavailable fields clearly labeled."),
        ("E4. Full combined matrix", "Does noise have the same practical effect far from and near the threshold?", "A noise level that produces small fluctuations far inside the locked regime may produce substantial escapes near threshold.", "Execute all 8×9 combinations, preserving each condition's configuration, raw trajectory rows, seeds, event list and sampled ensemble arrays.", "Complete 72-cell tables and heatmaps; comparisons are descriptive and retain unequal 500/1000 sample sizes."),
        ("E5. Coupled convergence", "Are step-size errors small and decreasing for both a mild baseline and a nonlinear stress condition?", "Additive-noise explicit integration can show approximately first-order strong convergence in a resolved regime, but the slope must be measured rather than imposed.", "Generate the finest Brownian path once per replicate, sum contiguous increments for each coarse h, compare at matched times and keep all paired endpoints/errors. Use a finite numerical reference at h=0.00015625, not an exact stochastic solution.", "Endpoint state RMS error with bootstrap intervals, paired phase and phase-square mean differences, variance differences, maximum sampled-path error and finite-output fractions."),
        ("E6. Completed slips and selected long horizons", "Do event rates survive reasonable changes in core/dwell definitions, and which rare events are resolved by a longer observation?", "Transient recrossings will inflate a naive crossing count. Completed-transition rates should be evaluated against paired loose/strict rules and censoring.", "Primary: adjacent well core radius π/4, dwell 0.5; loose: π/2,0.25; strict: π/6,1. Compare all three on each path. Independently run six n=1000,T=200 cohorts. Plot approximate deterministic saddle-manifold branches as context.", "Forward/backward counts, total completed-transition exposure rate, event probability, survival curve, restricted mean, observed/censored counts and rule sensitivity."),
        ("E7. Mechanistic a/b sensitivity", "How do a and b modify capture, dispersion and escape at fixed physical detuning?", "Changing a affects damping and effective noise gain; changing b affects restoring stiffness and the equilibrium threshold. Their combined effect need not be monotone in a single scalar.", "Cross a,b∈{0.5,1,2} and α∈{0.15,0.30}, fixing Δ=0.5, n=500,T=50,h=0.001. Classify regimes by each cell's actual δ=Δ/(ab).", "Structured 18-cell table, mean velocity/rotation, capture and slip rate only when stable wells exist."),
    ]
    for title, question, hypothesis, procedure, metrics in experiments:
        content += f'<article class="experiment"><h3>{esc(title)}</h3>'+paragraph(f'<strong>Question.</strong> {esc(question)}')+paragraph(f'<strong>Preregistered hypothesis.</strong> {esc(hypothesis)}')+paragraph(f'<strong>Procedure and controls.</strong> {esc(procedure)}')+paragraph(f'<strong>Readouts.</strong> {esc(metrics)}')+'</article>'
    content += '<h3>Statistical estimands and their uncertainty</h3>'
    content += equation("mean", r"\bar Y=\frac1M\sum_{r=1}^M Y_r,\quad s_Y^2=\frac1{M-1}\sum_{r=1}^M(Y_r-\bar Y)^2,\quad \bar Y\pm t_{0.975,M-1}\frac{s_Y}{\sqrt M}",
        "E1. Mean, sample variance and pointwise 95% mean interval", [
            ["M", "Number of independent complete trajectories in one condition."],
            ["r", "Trajectory index, not time-step index."],
            ["Yᵣ", "One scalar path outcome, or one state coordinate at a fixed common sample time."],
            ["Ȳ,s²ᵧ,sᵧ", "Sample mean, unbiased sample variance and sample standard deviation."],
            ["t₀.₉₇₅,ₘ₋₁", "Student-t quantile for a two-sided 95% mean interval with M−1 degrees of freedom."],
        ], "Repeated independent experiments would generate different sample means. The interval quantifies uncertainty in that mean. It is not a region expected to contain 95% of sample paths, nor a simultaneous guarantee over every plotted time.",
        '<code>mean_ci</code> implements this calculation; <code>experiment_runner</code> applies the same formula at sampled ensemble times. Single deterministic controls have no estimated Monte Carlo confidence interval.')
    content += equation("wilson", r"\widehat p=\frac{x}{M},\quad c=\frac{\widehat p+z^2/(2M)}{1+z^2/M},\quad q=\frac{z\sqrt{\widehat p(1-\widehat p)/M+z^2/(4M^2)}}{1+z^2/M},\quad \mathrm{CI}=[c-q,c+q]",
        "E2. Wilson confidence interval for a binary path outcome", [
            ["x", "Number of trajectories satisfying capture, retained lock or at-least-one-slip."],
            ["M,p̂", "Independent trajectory denominator and observed success fraction."],
            ["z", "Standard-normal 97.5th percentile for a two-sided 95% interval."],
            ["c,q", "Wilson center and half-width; endpoints are restricted to [0,1]."],
        ], "Observing no events does not make the population event probability known to be zero. The score interval remains nontrivial at zero or all observed successes, where a simple normal interval can be misleading.",
        '<code>wilson_interval</code> is applied to independent path indicators. Unlocked cells retain unavailable capture/slip fields instead of being assigned a misleading zero.')
    content += equation("sliprate", r"\widehat\nu=\frac{\sum_{r=1}^M C_r}{MT},\qquad P(\text{at least one slip})\ \text{is a different estimand},\qquad \nu_{\mathrm{upper}}\approx\frac{-\log(0.05)}{MT}\ \ (\text{zero events})",
        "E3. Completed-transition rate and a zero-event caution", [
            ["Cᵣ", "Total forward plus backward primary committed transitions on path r."],
            ["T", "Observation horizon shared by the paths in that condition."],
            ["ν̂", "Completed transitions per model time unit, averaged over independent paths."],
            ["MT", "Total path exposure; it is not an independent-time-step sample count."],
            ["νupper", "A labeled one-sided 95% Poisson exposure approximation when no event is observed."],
        ], "A path can have multiple transitions, so event probability and transition frequency answer different questions. Bootstrap intervals resample complete path counts, preserving their within-path dependence. With all counts zero the bootstrap interval collapses to zero; the Poisson bound adds an explicit modeling approximation rather than pretending zero uncertainty.",
        '<code>summarize</code> divides the path-count mean by T, bootstraps trajectory counts and records the separate <code>zero_event_rate_upper_poisson</code>. Wilson bounds for the at-least-one-event probability provide a direct independent-path uncertainty statement without a Poisson-process assumption.')
    content += equation("survival", r"\widehat S(t)=\prod_{t_j\le t}\left(1-\frac{d_j}{n_j}\right),\qquad \widehat{\mathrm{RMST}}(T)=\int_0^T\widehat S(t)\,dt",
        "E4. First-slip survival and restricted mean time", [
            ["S(t)", "Probability that the first committed slip occurs after time t."],
            ["tⱼ", "A distinct observed first-commitment time."],
            ["dⱼ,nⱼ", "Events at tⱼ and trajectories still at risk immediately before tⱼ."],
            ["RMST(T)", "Expected first-slip time truncated at observation horizon T, estimated by the survival area."],
            ["Right censoring", "A no-slip path is known to survive through T, but its eventual waiting time is unknown."],
        ], "No-slip trajectories carry information and remain at risk until the horizon. If fewer than half the paths slip, the median may not be reached. A restricted mean equal to T means no observed event within that window; it does not establish an infinite waiting time.",
        '<code>kaplan_meier</code> updates the risk set at observed first events and integrates the step survival function. The report shows censor counts, restricted means and unreached medians explicitly. Event times include the detector’s required residence interval.')
    content += equation("strong", r"e_{\mathrm{RMS}}(h)=\left[\frac1M\sum_{r=1}^M\left\|X_T^{(h,r)}-X_T^{(h_{ref},r)}\right\|_2^2\right]^{1/2},\qquad \widehat p=\operatorname{slope}(\log h,\log e_{\mathrm{RMS}})",
        "E5. Coupled endpoint strong error and a finite-range rate", [
            ["Xᵀ⁽ʰ,ʳ⁾", "Numerical phase–velocity endpoint for path r and step h."],
            ["href", "Finite reference step 0.00015625; reference and coarse path share Brownian forcing."],
            ["‖·‖₂", "Euclidean state norm; combines phase and velocity in model coordinates."],
            ["eRMS(h)", "Across-path root mean square endpoint state difference from the reference."],
            ["p̂", "Regression slope across the tested mesh sizes; an empirical diagnostic."],
        ], "Strong error compares paired realizations, not unrelated random paths. The state norm mixes coordinate units, so it is a defined computational error metric, not a calibrated physical distance. A finite reference biases the smallest errors; the fitted rate is not an exact-order theorem.",
        '<code>convergence_case</code> coarsens each fine path by integer sums, stores paired endpoints and bootstraps the RMS. Paired phase mean, second-moment and variance differences assess weak quantities without declaring a noisy weak-convergence slope.')
    content += '<h3>The synthetic extension: an explicitly separate model</h3>'
    content += equation("network", r"d\theta_i=\left[\omega_i+\frac{K}{d_i}\sum_j A_{ij}B_{ij}(t)\sin\!\left(\theta_j(t-\tau)-\theta_i(t-\tau)\right)\right]dt+\sigma\!\left(\sqrt{1-\rho}\,dW_i+\sqrt\rho\,dW_c\right)",
        "E6. Delayed-relative-error oscillator network", [
            ["i,j,N", "Receiving node, sending node and total number of phase oscillators."],
            ["θᵢ,ωᵢ", "Unwrapped node phase and fixed intrinsic frequency in model units."],
            ["K", "Coupling strength, fixed at 2 in this study."],
            ["Aᵢⱼ,dᵢ", "Adjacency matrix and original receiving-row degree; dropout does not renormalize this degree."],
            ["Bᵢⱼ(t)", "Stepwise Bernoulli received-message indicator, with dropout probability 0 or 0.3."],
            ["τ", "Delay 0 or 0.2 applied to the entire relative phase error; constant prehistory is used."],
            ["σ,ρ", "Noise amplitude 0.35 and shared fraction 0, 0.5 or 1; off-diagonal increment covariance is σ²ρ dt."],
            ["Wᵢ,Wc", "Independent node Brownian motions and a common Brownian motion, mutually independent before mixing."],
        ], "Neighbors contribute sine corrections through received graph edges. Increasing ρ transfers variance from independent to common forcing without changing each node's total noise variance. Because both nodes in the delayed difference use the same time, common additive noise cancels from every relative phase. Any higher coherence at ρ=1 reflects removal of differential forcing, not an intrinsic common-noise synchronization theorem.",
        '<code>synthetic_agents.simulate_networks</code> evolves a global mean plus centered phases to preserve this cancellation algebraically. The full matrix has 144 cells: N∈{3,8,20}, four topology labels, three ρ values, two dropout rates and two delays, each with 500 network trajectories, T=50,h=0.01. Inputs are matched across conditions at a fixed N, giving 1500 unique input identities rather than 72,000 mutually independent realizations.')
    content += equation("order", r"R(t)=\left|\frac1N\sum_{j=1}^N e^{\mathrm{i}\theta_j(t)}\right|,\qquad D(t)=\frac{2}{N(N-1)}\sum_{i<j}\big(1-\cos(\theta_i-\theta_j)\big)=\frac{N}{N-1}\big(1-R^2(t)\big)",
        "E7. Coherence and circular disagreement", [
            ["R(t)∈[0,1]", "Length of the mean unit phasor: one is perfect circular alignment."],
            ["exp(iθⱼ)", "Unit vector for node phase; i in the exponential is the imaginary unit."],
            ["D(t)", "Mean 1−cos phase difference over unordered node pairs."],
            ["N", "Node count; the N/(N−1) factor corrects exclusion of self-pairs."],
            ["Tail summaries", "Per-network averages over t≥40, followed by cross-network mean/SD/median and confidence intervals."],
        ], "Aligned vectors reinforce, dispersed vectors cancel. R can be small for several organized clusters and cannot by itself identify every network structure. D is an exact function of R² for this definition, so it is not an independent second proof. Frequency-dispersion summaries use tail net velocities, not an ill-defined derivative of white noise.",
        '<code>order_parameter</code>, <code>pairwise_disagreement</code> and the compiled synthetic kernel implement these quantities. <code>summarize_networks</code> treats each completed network trajectory as a replicate. N=3 complete and ring are the same undirected graph and remain labeled as duplicate topology evidence.')
    content += '''<h3>Optional LLM research: decision and approval gate</h3><p>The source-checked literature supports testing sparse debate topologies, but does not validate a phase mapping from language-model messages. No paid LLM experiment is executed here. A defensible proposal first tests whether an explicitly defined circular signal predicts answer revision, subsequent disagreement or correctness beyond ordinary semantic and answer disagreement. Tune that signal only on separate development data.</p><p>The official GSM8K files were downloaded and counted: <strong>7473 training and 1319 test problems</strong>, with file hashes preserved in the dataset manifest. That is dataset preparation, not benchmark inference. The proposed base comparison has five methods and seven logical calls per method/item, yielding <strong>46,165 logical calls</strong> for one full-test-split seed. Planning assumptions of 2000 input and 800 output tokens per call imply <strong>92.33 million input and 36.932 million output tokens</strong>. These are hypothetical workloads, not observed usage. No dollar total is defensible without an identified deployment, its prices and quotas. A nine-condition robustness cross and five seeds would multiply that workload by 45.</p><p>If the representation gate succeeds, retain frozen development decisions, hide all gold answers from model prompts, and compare independent agents, full communication, fixed sparse communication, disagreement-triggered communication and the proposed controller with identical roles/prompts/models. Record accuracy, tokens, messages, latency, cost and robustness, and use call-matched comparisons where feasible. Credentials stay in a secure local environment. Complete deployment-specific cost estimation and the user's approval are prerequisites to API calls. The complete conditional assessment is embedded in Section J.</p>'''
    return content


def interval(row, field, low=None, high=None, digits=4):
    """Display one estimate beside the actual stored interval, without filling missing bounds."""
    estimate = number(row.get(field), digits)
    lower, upper = row.get(low or field+"_low"), row.get(high or field+"_high")
    if lower is None or upper is None:
        return estimate + (" [no Monte Carlo CI]" if estimate != "not applicable" else "")
    return f"{estimate} [{number(lower,digits)}, {number(upper,digits)}]"


def result_table(name, rows, fields, title, labels=None):
    """Render complete measured rows and attach an offline CSV-download action."""
    labels = labels or {}
    output = table([labels.get(field,field.replace("_"," ")) for field in fields],
                   [[number(row.get(field)) for field in fields] for row in rows], title)
    return output + f'<button type="button" class="download" data-download="{esc(name)}">Download {esc(name.replace("_"," "))} CSV</button>'


def scientific_figure(path, captions, index):
    """Embed an actual chart with observation, interpretation and limitation captions."""
    source = captions[path.name]
    if isinstance(source, str):
        source = {"title":path.stem.replace("_"," "),"explanation":source,
                  "interpretation":"Read the plotted quantities under the definitions in Section E.",
                  "limitations":"The numerical and statistical scope is restricted to the recorded configuration."}
    caption = (f'<strong>Figure F{index}. {esc(source["title"])}.</strong> '
               f'{esc(source.get("explanation",""))} '
               f'<span><strong>Interpretation.</strong> {esc(source.get("interpretation",""))}</span>'
               f'<span><strong>Scope and limits.</strong> {esc(source.get("limitations",""))}</span>'
               f'<span class="source-ref">Embedded source: <code>results/figures/{esc(path.name)}</code>; '
               f'SHA-256 <code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code>.</span>')
    return f'<figure class="measured" id="figure-{esc(path.stem)}"><img loading="lazy" src="{data_uri(path)}" alt="{esc(source["title"])}"/><figcaption>{caption}</figcaption></figure>'


def interactive_plots():
    """Provide accessible offline controls whose series are loaded from executed JSON/CSV."""
    return '''<article class="interactive"><h3>Read the actual baseline acquisition series</h3>
    <p>This control draws the saved original-initial-state acquisition outputs. Change the readout to distinguish one noisy path, an estimated mean, a mean confidence interval and an across-path variance. The plot is generated from the embedded measured JSON, with no server or chart CDN.</p>
    <div class="controls"><label for="baseline-view">Readout <select id="baseline-view"><option value="compare">Deterministic, stochastic and ensemble mean phase</option><option value="mean">Ensemble mean phase with 95% mean CI</option><option value="variance">Across-path phase variance</option></select></label>
    <label for="baseline-time">Inspect sample time <input id="baseline-time" type="range" min="0" max="1000" value="1000"/></label></div>
    <svg id="baseline-interactive" viewBox="0 0 1000 410" role="img" aria-label="Actual measured baseline time series"></svg>
    <output id="baseline-readout" aria-live="polite"></output>
    <p class="note">A mean confidence interval describes uncertainty in the estimated mean. A single stochastic path can lie outside it without contradiction. Variance has units rad²; phase has units rad.</p></article>
    <article class="interactive"><h3>Inspect all 72 measured noise–detuning cells</h3><p>Select a quantity, then focus or click a cell for its parameter values, trajectory count and available uncertainty. Blank cells mean the requested locked-well observable is not applicable, rather than a measured zero.</p>
    <div class="controls"><label for="grid-metric">Quantity <select id="grid-metric"><option value="phi_variance">Terminal phase variance (rad²)</option><option value="locking_probability">Terminal capture probability</option><option value="retained_lock_probability">Retained-lock probability</option><option value="slip_rate">Completed-slip rate (transitions/time)</option><option value="tail_velocity_mean">Tail mean velocity (rad/time)</option><option value="rotation_rate">Tail rotation rate (cycles/time)</option></select></label></div>
    <div id="grid-interactive" class="grid-scroll"></div><output id="grid-readout" aria-live="polite">Select a measured cell to inspect its denominator and estimate.</output>
    <p class="note">Every stochastic cell has 500 or 1000 independent paths; α=0 is a single deterministic control. The color scale is recomputed for the selected quantity and is descriptive. Numerical tables below retain all uncertainty fields.</p></article>'''


def results(root, metadata, tables, figures, captions):
    """Compose full results from executed cohorts, including unavailable and zero-event outcomes."""
    grid = tables["combined_summary"]
    original = read_csv(root / "results/tables/original_initial_summary.csv")
    tables["original_initial_summary"] = original
    effects_path = root / "results/tables/effect_sizes.csv"
    if effects_path.exists():
        tables["effect_sizes"] = read_csv(effects_path)
    deterministic_path = root / "results/tables/deterministic_validation.csv"
    if deterministic_path.exists():
        tables["deterministic_validation"] = read_csv(deterministic_path)
    paired_path = root / "experiments/synthetic_agents/paired_effects.csv"
    if paired_path.exists():
        tables["synthetic_paired_effects"] = read_csv(paired_path)
    lookup = {path.name:path for path in figures}
    figure_index = 0
    def show(names):
        nonlocal figure_index
        output = ""
        for name in names:
            if name in lookup:
                figure_index += 1
                output += scientific_figure(lookup.pop(name),captions,figure_index)
        return output
    content = '''<p>Every scalar in this section is loaded from an executed summary file. Saved charts are embedded without editing their numerical content. The complete tables retain measured zeros, unavailable unlocked-state quantities, deterministic controls without Monte Carlo intervals and all censoring fields. The conceptual diagrams in Section C are not substituted for these results.</p>'''
    content += '<h3>F1. Baseline: equilibrium fluctuations and original acquisition are different cohorts</h3>'
    base = next(row for row in grid if row["delta"]==.5 and row["alpha"]==.15)
    original_row = original[0]
    content += table(["Cohort", "n", "Terminal mean φ (rad), 95% CI", "Terminal variance φ (rad²), bootstrap CI", "Capture probability, Wilson CI", "Committed slip rate, bootstrap CI"], [
        ["Stable-phase grid initial state",number(base["n"]),interval(base,"phi_mean"),interval(base,"phi_variance"),interval(base,"locking_probability"),interval(base,"slip_rate")],
        ["Original displaced initial phase",number(original_row["n"]),interval(original_row,"phi_mean"),interval(original_row,"phi_variance"),interval(original_row,"locking_probability"),interval(original_row,"slip_rate")],
    ], "Two separately executed 1000-path baseline cohorts. All times are model units.")
    content += paragraph(f'The stable-phase baseline has terminal mean velocity {interval(base,"velocity_mean")} rad/time and tail mean velocity {interval(base,"tail_velocity_mean")} rad/time. The displaced-state cohort is used to teach acquisition because it begins 0.5 rad away from the equilibrium. Similar terminal outcomes would not erase that difference in initial conditions.')
    replay_path = root / "experiments/baseline/original_replay.json"
    replay = json.loads(replay_path.read_text())
    content += table(["Original seed", "Maximum phase difference (rad)", "Maximum velocity difference (rad/time)"],
                     [[row["seed"],number(row["max_phase_difference"],8),number(row["max_velocity_difference"],8)] for row in replay],
                     "Exact original-code replay on identical Brownian increments and original initial conditions")
    content += '<p>The preserved baseline already uses the correct simultaneous explicit update. The replay checks numerical identity rather than claiming that a different initial condition or a replacement noise convention reproduces it. The modular implementation adds validation, statistics, completed-transition detection and a complete experiment design.</p>'
    if "deterministic_validation" in tables:
        content += '<h4>Zero-noise agreement with an independent ODE solver</h4>'
        content += result_table("deterministic_validation",tables["deterministic_validation"],["delta","dt","T","max_state_error","time_of_max_error","endpoint_state_error"],"Euler–Maruyama with α=0 versus SciPy DOP853, rtol=1e−11 and atol=1e−13")
        content += '<p>These deterministic mesh comparisons are numerical checks, not extra independent stochastic cohorts. A small endpoint difference after damping can conceal a larger earlier transient error, so the maximum state error and its time are reported. The two detunings use the stable phase plus a 0.5-rad displacement.</p>'
    content += show(["baseline.svg", "brownian.svg"])
    content += interactive_plots()
    content += '<h3>F2. Noise sensitivity at fixed δ=0.5</h3>'
    noise = [row for row in grid if row["delta"]==.5]
    content += table(["α", "n", "Terminal variance, 95% CI (rad²)", "Mean absolute circular phase deviation (rad)", "Capture probability, 95% CI", "Slip rate, 95% CI"],
                     [[number(row["alpha"]),number(row["n"]),interval(row,"phi_variance"),number(row.get("mean_absolute_phase_deviation")),interval(row,"locking_probability"),interval(row,"slip_rate")] for row in noise],
                     "E2 is the fixed-detuning slice of the same grid; it is not an additional independent ensemble.")
    content += '<p>At this detuning the distinction between spread and escape matters. A terminal variance can increase while the completed-slip rate remains unresolved at zero within T=50. A larger circular error is not itself a full 2π transition. Conversely, unwrapped variance after escapes mixes within-well fluctuation with separation among occupied wells.</p>'
    content += '<h3>F3. Detuning and the full noise–detuning interaction</h3>'
    detuning = [row for row in grid if row["alpha"]==.15]
    content += table(["δ", "n", "Terminal phase variance, 95% CI", "Tail velocity, 95% CI (rad/time)", "Capture probability, 95% CI", "Slip rate, 95% CI", "Rotation (cycles/time)"],
                     [[number(row["delta"]),number(row["n"]),interval(row,"phi_variance"),interval(row,"tail_velocity_mean"),interval(row,"locking_probability"),interval(row,"slip_rate"),number(row.get("rotation_rate"))] for row in detuning],
                     "E3 at α=0.15. Above threshold, capture and ordinary locked-well slips are not applicable.")
    content += show(["noise_detuning.svg", "combined_grid.svg", "potential_barrier.svg"])
    content += paragraph('The interaction is a finite-horizon observation under the declared initial states and noise convention. The barrier and local eigenvalues provide a mechanism for increased susceptibility near the threshold; the measured rate and capture estimates come from the paths. A nonzero rotation rate is transport, not evidence that the integrator diverged.')
    content += '<details><summary>Inspect all 72 grid rows, including interval endpoints, medians, SDs, censoring and zero-event upper bounds</summary>'
    content += result_table("combined_summary", grid, list(grid[0]), "Complete executed grid; source CSV field names retain exact definitions") + '</details>'
    content += '<h3>F4. Descriptive between-condition effect sizes</h3>'
    if "effect_sizes" in tables:
        effects = tables["effect_sizes"]
        content += result_table("effect_sizes",effects,["comparison","metric","n_left","n_right","difference","ci_low","ci_high","standardized_difference"],
                                "Selected independent-cohort differences with 2000-resample bootstrap intervals")
        content += '<p>The comparison label states the direction of subtraction; the condition IDs preserve the exact cohorts. The difference is in the metric’s original units; the standardized difference divides by a pooled between-trajectory SD where defined. A binary all-equal comparison can have zero bootstrap width. These contrasts were selected after inspecting the outputs and are <strong>exploratory and descriptive</strong>. They are not preregistered tests or familywise significance declarations, and their interpretation does not depend only on whether an interval excludes zero.</p>'
    content += '<h3>F5. Selected long horizons, first-slip censoring and detector sensitivity</h3>'
    long_rows = tables["long_horizon_summary"]
    content += table(["δ", "α", "n / T", "Events / censored first slips", "Primary rate, 95% CI", "Loose / strict rate", "RMST to T", "First-slip median"],
                     [[number(row["delta"]),number(row["alpha"]),f'{number(row["n"])} / {number(row["T"])}',f'{number(row.get("first_slip_events"))} / {number(row.get("first_slip_censored"))}',interval(row,"slip_rate"),f'{number(row.get("loose_slip_rate"))} / {number(row.get("strict_slip_rate"))}',number(row.get("rmst_first_slip")),number(row.get("median_first_slip"))] for row in long_rows],
                     "Six independently executed T=200 cohorts. An unavailable median means the survival curve did not reach 0.5.")
    zero = [row for row in long_rows if row.get("slips_total")==0]
    if zero:
        content += table(["δ / α", "Observed rate", "Wilson upper bound: ≥1 event probability", "Poisson exposure upper rate approximation"],
                         [[f'{number(row["delta"])} / {number(row["alpha"])}',"0",number(row.get("slip_probability_high")),number(row.get("zero_event_rate_upper_poisson"))] for row in zero],
                         "Zero-event uncertainty. The probability bound is a path-level statement; the rate bound additionally assumes Poisson exposure.")
    content += show(["slip_survival.svg", "slip_geometry.svg"])
    content += '<p>The three detectors are evaluated on identical paths. Their differences measure sensitivity to what counts as completed arrival, not sampling uncertainty from three independent studies. Longer observation changes censoring and exposure; a T=200 rate should not be pooled blindly with T=50 if transient or nonstationary behavior matters. No surviving path is assigned an infinite first-slip time.</p>'
    content += '<details><summary>Inspect every long-horizon summary field</summary>' + result_table("long_horizon_summary",long_rows,list(long_rows[0]),"Complete long-horizon condition summaries") + '</details>'
    content += '<h3>F6. Numerical convergence on paired Brownian paths</h3>'
    conv = tables["convergence_summary"]
    content += table(["Scenario", "h", "n", "Endpoint state RMS, bootstrap 95% CI", "Paired phase-mean bias, 95% CI (rad)", "Variance bias, 95% CI (rad²)", "Finite paths"],
                     [[row["scenario"],number(row["dt"]),number(row["n"]),interval(row,"strong_rms","strong_low","strong_high"),interval(row,"mean_bias"),interval(row,"variance_bias"),number(row["finite_fraction"])] for row in conv],
                     "Each candidate mesh shares the same Brownian forcing as the finite reference h=0.00015625.")
    for scenario in sorted({row["scenario"] for row in conv}):
        rows = [row for row in conv if row["scenario"]==scenario]
        content += paragraph(f'For <strong>{esc(scenario.replace("_"," "))}</strong>, the log–log fitted endpoint state-RMS slope over the five tested steps is <strong>{number(rows[0].get("rate_fit"))}</strong>. The largest-step RMS is {number(max(rows,key=lambda r:r["dt"])["strong_rms"])} and the smallest-step RMS is {number(min(rows,key=lambda r:r["dt"])["strong_rms"])} in the defined state norm. This is an empirical rate against a numerical reference, with the tested horizon restricted to T=10.')
    content += show(["convergence.svg"])
    content += '<p>The variance-bias interval is computed by resampling matched endpoint rows and subtracting variances inside each resample. Mean and phase-square differences also preserve pairing. Weak differences that are small relative to their uncertainty do not support an additional claimed weak-order slope. T=10 convergence does not validate every rare-event count accumulated over T=200; a long-horizon event-refinement study remains useful.</p>'
    content += '<details><summary>Inspect all convergence fields, including second-moment bias and sampled-path error</summary>' + result_table("convergence_summary",conv,list(conv[0]),"Complete paired convergence diagnostics") + '</details>'
    content += '<h3>F7. Parameter sensitivity at fixed physical detuning</h3>'
    sensitivity = tables["sensitivity_summary"]
    content += result_table("sensitivity_summary",sensitivity,["a","b","delta","alpha","n","phi_variance","tail_velocity_mean","locking_probability","retained_lock_probability","slip_rate","rotation_rate"],
                            "All 18 a/b/noise conditions; Δ=0.5,T=50,h=0.001,n=500 per cell")
    content += show(["parameter_sensitivity.svg"])
    content += '<p>The normalized damping 1/(a√b) and noise α/(ab³ᐟ⁴) change along with δ when a or b is varied at fixed Δ and α. This study therefore reveals joint mechanisms rather than an isolated partial derivative at fixed dimensionless groups. Several parameter combinations have no equilibrium; their transport is interpreted as rotation and their locked-well columns remain unavailable.</p>'
    content += '<details><summary>Inspect parameter-study confidence intervals, medians and event/censoring fields</summary>' + result_table("sensitivity_summary",sensitivity,list(sensitivity[0]),"Complete parameter sensitivity summaries") + '</details>'
    content += '<h3>F8. Synthetic oscillator networks: measured extension and exact model limits</h3>'
    synth = tables["synthetic_summary"]
    content += paragraph(f'The synthetic study executed <strong>{len(synth)} cells and {sum(int(row["n"]) for row in synth):,} network condition-runs</strong>. Each cell has 500 independent network trajectories. Inputs are deliberately matched across conditions at each network size, so there are <strong>1500 independent matched network-input blocks</strong>, not 72,000 mutually independent input realizations. The N=3 complete and ring labels denote the same graph.')
    selected = [row for row in synth if row.get("dropout")==0 and row.get("delay")==0 and row.get("topology") in {"complete","ring","sparse"}]
    content += table(["N", "Topology", "ρ", "n", "Tail mean R, 95% CI", "Tail disagreement, 95% CI", "Frequency dispersion, 95% CI"],
                     [[number(row.get("n_agents")),row.get("topology"),number(row.get("rho")),number(row.get("n")),interval(row,"tail_R_mean_mean","tail_R_mean_ci_low","tail_R_mean_ci_high"),interval(row,"tail_pairwise_disagreement_mean","tail_pairwise_disagreement_ci_low","tail_pairwise_disagreement_ci_high"),interval(row,"tail_frequency_dispersion_mean","tail_frequency_dispersion_ci_low","tail_frequency_dispersion_ci_high")] for row in selected],
                     "Selected unimpeded synthetic cells; complete 144-cell table follows. CIs use network-level tail aggregates.")
    content += show(["synthetic_synchronization.svg","synthetic_impairments.svg","synthetic_timeseries.svg"])
    content += '<p>The common additive noise cancels from relative phases under the stated stale-relative-error model. At ρ=1, differences between networks arise from their initial phases and fixed frequency/coupling conditions rather than independent differential Brownian forcing. Dropout reduces the expected coupling because the original degree is retained in the denominator; its observed effect includes that weakening and cannot be called a pure message-efficiency effect at equal mean coupling.</p>'
    ring = sorted([row for row in synth if row.get("n_agents")==8 and row.get("topology")=="ring" and row.get("dropout")==0 and row.get("delay")==0],key=lambda row:row["rho"])
    content += '<h4>An unexpected result: frequency locking without high global phase alignment</h4>'
    content += table(["8-node clean ring, ρ", "Tail mean R, 95% CI", "Tail frequency dispersion, 95% CI"],[[number(row["rho"]),interval(row,"tail_R_mean_mean","tail_R_mean_ci_low","tail_R_mean_ci_high"),interval(row,"tail_frequency_dispersion_mean","tail_frequency_dispersion_ci_low","tail_frequency_dispersion_ci_high")] for row in ring],"Increasing the shared fraction does not monotonically increase the observed global coherence in this ring cohort.")
    diagnostic_path=root/'experiments/synthetic_agents/ring_state_diagnostic.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        content += table(["Selected original replicate", "Tail R", "Frequency dispersion", "Terminal ring winding", "Replay R error"],[[number(row["original_replicate"]),number(row["tail_R"]),number(row["frequency_dispersion"]),number(row["terminal_winding"]),number(row["relative_metric_replay_absolute_error"])] for row in diagnostic["records"]],"Exact relative-dynamics replays of two previously executed ρ=1 ring paths; zero additional independent networks.")
        content += paragraph('The low-R example has an approximately −1 winding around the ring while its node frequencies are nearly locked. A phase-twisted configuration can cancel unit phasors even when velocities agree. Winding here is the sum of principal edge phase gaps divided by 2π, with edges traversed once around the ring. The two examples were selected from the executed cohort, preserve their original inputs and match the recorded tail R exactly; they illustrate two observed states without classifying every one of the 500 paths.')
        content += '<details><summary>Inspect the selected-ring diagnostic and original inputs</summary><pre class="plain-source"><code>'+esc(json.dumps(diagnostic,indent=2))+'</code></pre></details>'
    if "synthetic_paired_effects" in tables:
        positive = [row for row in tables["synthetic_paired_effects"] if row.get("n_agents")==8 and row.get("topology")=="ring" and row.get("rho")==1 and row.get("dropout")==.3 and row.get("delay")==0 and row.get("metric")=="tail_R_mean"]
        if positive:
            row=positive[0]
            content += paragraph(f'For this same 8-node, ρ=1 ring, adding dropout 0.3 changes mean tail R by {number(row["paired_difference_mean"])} with paired 95% interval [{number(row["ci_low"])}, {number(row["ci_high"])}]. Thus dropout is not uniformly harmful to this particular global-coherence metric. Random masking and weaker expected coupling change together, and the result does not establish a unique causal mechanism or an LLM communication benefit.')
    if "synthetic_paired_effects" in tables:
        content += '<details><summary>Inspect paired impairment-minus-baseline differences for the synthetic networks</summary>' + result_table("synthetic_paired_effects",tables["synthetic_paired_effects"],list(tables["synthetic_paired_effects"][0]),"Matched-network descriptive effect intervals; no multiplicity-adjusted significance claims") + '</details>'
    content += '<details><summary>Inspect the complete synthetic matrix with means, SDs, medians and network confidence intervals</summary>' + result_table("synthetic_summary",synth,list(synth[0]),"All executed synthetic condition rows") + '</details>'
    content += '<h3>F9. Unexpected outcomes, implementation limits and missing claims</h3><p>The initial attempt to use a process pool was blocked by the execution sandbox before any condition ran. A bounded thread pool was used instead; the ctypes kernel releases the Python GIL during compiled work. This is an execution-environment change, not a reduced matrix. Exact replay and solver parity protect the numerical interpretation.</p><p>Zero-event cohorts and unreached first-slip medians are retained as outcomes. Their unresolved tails are not filled by an asymptotic formula. No numerical “Milstein improvement” is reported because its correction vanishes. No language-model accuracy, API cost or communication-efficiency gain is inferred from a network order parameter.</p>'
    if lookup:
        content += '<h3>Additional saved scientific figures</h3>' + show(list(lookup))
    return content


def inline_markdown(text):
    """Render the small trusted-note Markdown subset while escaping source content."""
    text = esc(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+(?:\([^)]*\)[^\s)]*)?)\)",r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',text)
    text = re.sub(r"`([^`]+)`",r"<code>\1</code>",text)
    text = re.sub(r"\*\*([^*]+)\*\*",r"<strong>\1</strong>",text)
    return text


def note_html(path):
    """Embed actual Markdown evidence notes without requiring a Markdown runtime."""
    if not path.exists():
        raise FileNotFoundError(f"Required scientific note is absent: {path}")
    lines = path.read_text().splitlines()
    output, paragraphs, i = [], [], 0
    def flush():
        if paragraphs:
            output.append(paragraph(inline_markdown(" ".join(paragraphs))))
            paragraphs.clear()
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush(); code=[]; i+=1
            while i<len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i+=1
            output.append('<pre class="plain-source"><code>'+esc("\n".join(code))+'</code></pre>')
        elif line.strip() in {r"\[", "$$"}:
            flush(); end=r"\]" if line.strip()==r"\[" else "$$"; math_lines=[]; i+=1
            while i<len(lines) and lines[i].strip()!=end:
                math_lines.append(lines[i]);i+=1
            latex=re.sub(r"\\tag\{[^}]*\}","","\n".join(math_lines))
            output.append('<div class="math-display">\\['+esc(latex)+'\\]</div>')
        elif re.match(r"^#{1,6}\s",line):
            flush();level=min(5, len(line)-len(line.lstrip("#"))+2)
            output.append(f'<h{level}>'+inline_markdown(line.lstrip("#").strip())+f'</h{level}>')
        elif line.strip().startswith("|") and i+1<len(lines) and re.match(r"^\s*\|?[\s:|-]+\|\s*$",lines[i+1]):
            flush();headers=[part.strip() for part in line.strip().strip("|").split("|")];rows=[];i+=2
            while i<len(lines) and lines[i].strip().startswith("|"):
                rows.append([inline_markdown(part.strip()) for part in lines[i].strip().strip("|").split("|")]);i+=1
            output.append(table(headers,rows,raw=True));i-=1
        elif not line.strip():
            flush()
        elif re.match(r"^\s*(?:[-*]|\d+\.)\s+",line):
            flush();items=[]
            ordered=bool(re.match(r"^\s*\d+\.\s+",line))
            while i<len(lines) and re.match(r"^\s*(?:[-*]|\d+\.)\s+",lines[i]):
                items.append('<li>'+inline_markdown(re.sub(r"^\s*(?:[-*]|\d+\.)\s+","",lines[i]))+'</li>');i+=1
            tag="ol" if ordered else "ul"; output.append(f'<{tag}>'+"".join(items)+f'</{tag}>');i-=1
        else:
            paragraphs.append(line.strip())
        i+=1
    flush()
    return "".join(output)


def literature(root):
    """Use the primary-source-verified literature note and disclose its review scope."""
    return ('<p>This review distinguishes a verified bibliographic record, an accessible abstract and an inspected full text. A source is not credited with a theorem or result that was not checked. The review is targeted to this project; it is not an exhaustive claim about every publication through 2026. Publisher, DOI, ACL and official repository links remain useful when online, while every explanation in this report remains readable offline.</p>'
            + note_html(root / "notes/literature_review.md")
            + '<details><summary>Inspect the actual bibliography file</summary><pre class="plain-source"><code>'
            + esc((root / "report/references.bib").read_text()) + '</code></pre></details>')


def discussion(tables):
    """Separate observations, mechanisms and unresolved numerical/statistical questions."""
    grid=tables["combined_summary"]
    below=[row for row in grid if row["delta"]<1 and row["alpha"]>0]
    zero_count=sum(row.get("slips_total")==0 for row in below)
    maximum=max(below,key=lambda row:row.get("slip_rate") or 0)
    content=paragraph(f'Of the {len(below)} stochastic grid cells with stable wells, {zero_count} recorded no primary committed transition within T=50. The largest observed completed-slip rate among these cells was {interval(maximum,"slip_rate")} transitions/time at δ={number(maximum["delta"])}, α={number(maximum["alpha"])}. These statements retain the finite horizon and detector definition; they do not identify a sharp stochastic locking threshold.')
    issues=[
        ("Mechanism versus measured consequence", "The local restoring slope b cosφₛ and downhill barrier both weaken near the saddle-node. That mechanism makes the observed parameter interaction intelligible. It does not specify an exact escape distribution or a hardware lock range. The empirical matrix supplies the finite-horizon outcomes under its own initial conditions."),
        ("Capture is local and operational", "A stable deterministic equilibrium exists below threshold, but an inertial loop can have state-dependent acquisition. The grid deliberately starts near a well; the separate original-state cohort tests one displacement, not all basins. Nonzero additive noise also means the stable point is not an absorbing stochastic state."),
        ("Variance is more than one phenomenon", "Across-path unwrapped terminal variance combines intrawell spread and differing well indices after slips. Circular errors suppress whole-cycle differences and answer a different question. Within-path tail variance describes time variation within one path. Comparing them without their definitions can reverse the interpretation."),
        ("Finite numerical reference", "Coupled errors decrease on the tested meshes, but the reference is finite-step Euler–Maruyama, not an exact stochastic solution. The smallest-step errors partly reflect reference bias. The reported state norm combines coordinates in model units; no physical calibration makes it a universal distance."),
        ("Event accuracy at long horizons", "A convergence study over T=10 verifies trajectory refinement under two regimes. It does not alone certify a discontinuous dwell/escape functional over T=200. A follow-up should couple long paths at several dt values and compare event-time/count changes under fixed detector rules."),
        ("Detector geometry", "Core-and-dwell commitment prevents brief recrossings from inflating counts, but its radii and residence times are operational choices. The unstable saddle's stable manifold is two-dimensional context. The computed local branch is approximate, and neither that plot nor a phase-only boundary proves an exact stochastic basin-crossing rule."),
        ("Censoring and unresolved rare events", "Zero-event bootstraps collapse because resampling cannot invent an unobserved event. Wilson bounds for path event probability remain informative; Poisson exposure bounds require an additional process approximation. Restricted mean time is truncated at T, and an unreached median is not a numerical failure or infinite waiting-time estimate."),
        ("Uncertainty and multiplicity", "Trajectory-level t, Wilson and bootstrap intervals are descriptive. They do not cover all plotted times simultaneously, and no familywise correction is applied to a large matrix of selected comparisons. A future confirmatory test should predeclare contrasts, sampling rules, family definitions and multiplicity control."),
        ("Synthetic pairing and duplicates", "There are 500 independent networks within each cell, but topology/noise/impairment cells use matched initial and random inputs. Aggregate cell counts are condition-runs, not independent realizations. N=3 ring and complete are the same graph. Paired differences should be analyzed within those matched blocks."),
        ("Common noise and dropout interpretation", "Under the exact delayed-relative-state coupling, common additive noise cancels in relative phases. High coherence at ρ=1 therefore does not show that common noise actively contracts a phase difference. Dropout reduces effective mean coupling because the original degree remains fixed. An equal-mean-coupling comparison is needed to isolate stochastic message loss from average weakening."),
        ("Unvalidated physical assumptions", "The model idealizes forcing as Gaussian white noise and fixes the detuning and feedback law. Colored noise, frequency drift, multiplicative noise, saturating filters and hardware measurement errors are not included. Physical time and parameter units are not calibrated by circuit data."),
        ("Boundary of the LLM analogy", "Explicit oscillator phases have a natural 2π periodicity; language embeddings and answer strings do not automatically have one. Synthetic coherence is not answer accuracy. The candidate phase representation has not shown incremental predictive validity, so a paid full-split LLM experiment is not scientifically qualified by this local study alone."),
    ]
    for title,text in issues:
        content+=f'<h3>{esc(title)}</h3>'+paragraph(esc(text))
    return content


def conclusion(metadata,tables):
    """Summarize achieved local evidence and propose experiments that resolve specific limits."""
    conv=tables["convergence_summary"]
    slopes={row["scenario"]:row["rate_fit"] for row in conv}
    content=paragraph(f'The project completed the source-grounded model derivation, numerical validation and the full preregistered local PLL matrix: {number(metadata.get("independent_pll_trajectories"))} independent PLL path realizations across the grid, acquisition, long-horizon, parameter and paired-convergence cohorts. The synthetic study separately completed {len(tables["synthetic_summary"])} cells with matched network inputs. The report preserves their distinction and includes the current code, actual data summaries, saved figures and source-page provenance.')
    content+=paragraph('The important scientific lesson is the separation of equilibrium existence, finite-horizon capture, completed well transitions and continuous phase transport. Noise and detuning act through a weakening restoring landscape, while numerical and statistical definitions determine what can be claimed from finite paths. The constant diffusion also yields a concrete methodological conclusion: standard Milstein and Euler–Maruyama have exactly the same update for this SDE.')
    content+=paragraph('The observed coupled strong-error slopes were '+', '.join(f'{esc(name.replace("_"," "))}: {number(value)}' for name,value in slopes.items())+'. These are empirical finite-mesh results against a numerical reference. They support numerical refinement on the tested cases and horizon, while leaving rare-event functional convergence as a further question.')
    content+='''<h3>Follow-up experiments with clear decisions</h3><ol>
    <li><strong>Event-refinement validation.</strong> Reuse fine Brownian paths at T=200, coarsen them, and measure paired differences in primary/loose/strict commitment times and counts. This determines whether rare-event conclusions survive time-step refinement.</li>
    <li><strong>Dimensionless parameter isolation.</strong> Vary normalized detuning, damping and noise one at a time using the derived transformation. This separates effects that are jointly changed in the fixed-Δ a/b study.</li>
    <li><strong>Acquisition basins.</strong> Sample a preregistered phase–velocity initial-state grid, compare with resolved saddle-manifold geometry and retain noncapture/rotation. This addresses global acquisition more directly than stable-phase starts.</li>
    <li><strong>Rare-event precision.</strong> Choose a target event-probability or rate precision, calculate exposure requirements and extend independent paths under a fixed stopping rule. Importance sampling or splitting would require separate likelihood-weight validation.</li>
    <li><strong>Communication controls.</strong> Compare dropout with coupling rescaled to preserve its mean, test multiple sparse graph realizations, and refine dt while keeping the message-loss process physically consistent. Add sender-only delay only as an explicitly different model.</li>
    <li><strong>LLM representation gate.</strong> Establish a circular signal's held-out predictive value beyond ordinary disagreement before selecting a full benchmark plan. If the gate fails, study disagreement control on its own terms instead of forcing a physical-phase interpretation.</li></ol>
    <p>The paid LLM extension remains a conditional research design. No API inference, benchmark accuracy or monetary cost is reported as an observed result of this project.</p>'''
    return content


def reproducibility(root, metadata, tables, figures):
    """Expose actual provenance, environment, saved paths, commands and source evidence notes."""
    content='''<p>The HTML is a generated view of saved evidence. Open it directly from disk: images, scientific diagrams, syntax highlighting, measured series, controls and the MathJax SVG runtime are inside the file. Reading it requires no Python server. Re-execution does require the recorded Python environment, a C++17 compiler and the scientific packages used by the simulation scripts.</p>
    <h3>Execution commands and order</h3>
    <p>Run commands from the project root. Reusing checkpoints is valid only when configuration and numerical source identity match. The synthetic runner additionally validates hashes of its saved artifacts. A regeneration command consumes completed data; it does not rerun the experiments.</p>
    <pre class="plain-source"><code>python -m pytest tests -q
python -m src.experiment_runner --workers 2
python -m src.analysis_supplement
python -m src.visualization
python -m src.synthetic_agents --full
python -m src.verify_outputs
python -m src.deterministic_validation
python -m src.latex_report
python -m src.report_generator
tectonic --keep-logs report/main.tex

# Validated orchestration, including completion/tests/integrity gates:
python -m src.reproduce
# Rebuild the reports from existing completed evidence:
python -m src.reproduce --reports-only</code></pre>'''
    content+= '<h3>Actual run metadata</h3><pre class="plain-source"><code>'+esc(json.dumps(metadata,indent=2))+'</code></pre>'
    content+=table(["Evidence", "Saved location", "Meaning / reproduction"],[
        ["Preserved input implementation", "backups/baseline_1.py", "Original bytes and original figures retained; original_manifest.json records identity."],
        ["Realized full design", "configs/realized_design.json; configs/full_experiments.yaml", "Condition IDs, n, initial state, dt, horizon and parameter values."],
        ["PLL raw paths / events", "results/raw_data/condition_<id>/trajectories.csv; events.csv", "Independent path rows with uint64 seeds, summary metrics and committed events."],
        ["Ensemble sample arrays", "results/raw_data/condition_<id>/ensemble.npz", "Time, means, variances, pointwise mean intervals, representatives and terminal states."],
        ["Condition checkpoints", "results/raw_data/condition_<id>/summary.json; config.json", "Configuration fingerprint and numerical source hash; nested survival output retained."],
        ["Original replay", "experiments/baseline/original_replay.json; original_seed42.npz", "Six exact seeds compared to preserved baseline and the seed-42 path."],
        ["Paired convergence", "experiments/convergence/<scenario>_paired_raw.npz", "Seeds, candidate steps, errors, biases, endpoints and maximum sampled-path errors."],
        ["Analysis tables", "results/tables/*_summary.csv; effect_sizes.csv", "Exact data embedded into Section F; confidence endpoints, medians and SDs retained."],
        ["Synthetic raw evidence", "experiments/synthetic_agents/<cell>/", "Per-network rows, configuration/adjacency/frequencies, sampled states, checkpoint checksums."],
        ["Synthetic paired effects", "experiments/synthetic_agents/paired_effects.csv", "Within-input impairment-minus-baseline contrasts, preserving pairing."],
        ["Science figures", "results/figures/*.svg; captions.json", "Regenerated from saved outputs by visualization and synthetic figure routines."],
        ["Teaching diagrams", "src/report_generator.py: diagrams()", "Thirteen authored conceptual SVGs, explicitly not numerical evidence."],
        ["LLM assessment", "experiments/llm_extension/assessment.md; dataset_manifest.json", "Conditional full-split design, downloaded official dataset counts/hashes; no inference."],
        ["Environment and tests", "results/run_metadata/environment.json; *tests*.txt", "Recorded interpreter/packages/platform, test logs and initial failed preflight evidence."],
        ["Report asset identity", "report/report_assets.json", "Hashes and sizes of files embedded in this HTML."],
    ],"Artifact-to-claim map")
    content+='<h3>Randomness and reproducibility limits</h3><p>PLL trajectory seeds are derived with NumPy SeedSequence from the recorded master, condition ID and path ID. Within a cell the trajectories are independent; across convergence meshes the same Brownian increments are deliberately coupled. A seed without its generator, version, stream derivation, initial condition and dt is not a complete reproduction recipe.</p><p>The synthetic study records three stream identities per network. Its C++ Brownian generator is <code>std::mt19937_64</code> with <code>std::normal_distribution</code>; the latter is not required to be bitwise identical across standard-library implementations. Source, compiler/library environment, raw arrays and checksums therefore accompany the seeds. Dropout uses a counter-based keyed hash. Cell labels are omitted from synthetic stream derivation to produce matched input blocks.</p>'
    environment=root/'results/run_metadata/environment.json'
    content+='<details><summary>Inspect the actual recorded Python/platform/package environment</summary><pre class="plain-source"><code>'+esc(environment.read_text())+'</code></pre></details>'
    content+='<h3>Source hashes for this report</h3>'
    files=sorted((root/'src').glob('*.py'))+sorted((root/'src').glob('*.cpp'))
    content+=table(["Current source", "Bytes", "SHA-256"],[[str(path.relative_to(root)),str(path.stat().st_size),hashlib.sha256(path.read_bytes()).hexdigest()] for path in files],"Current source files read during report generation. The simulation's source identity is also retained separately in run metadata.")
    content+='<h3>Scientific notes and audit trail, embedded in full</h3>'
    notes=[("notes/project_audit.md","Original implementation audit and preservation"),
           ("notes/mathematical_model.md","Full source verification, derivation and mathematical scope"),
           ("notes/experiment_protocol.md","Preregistered PLL experiment protocol"),
           ("notes/numerical_validation.md","Numerical validation and observed limitations"),
           ("notes/synthetic_protocol.md","Separate synthetic protocol and matched-input semantics"),
           ("experiments/llm_extension/assessment.md","Conditional LLM research assessment and cost formula"),
           ("notes/report_design.md","Report design and verification responsibilities")]
    for relative,title in notes:
        content+=f'<details class="evidence-note"><summary>{esc(title)} <code>{esc(relative)}</code></summary>'+note_html(root/relative)+'</details>'
    for relative,title in [("results/run_metadata/preflight_tests.txt","PLL preflight tests"),("results/run_metadata/final_tests.txt","Final post-execution tests"),("results/run_metadata/original_manifest.json","Original source preservation manifest"),("experiments/llm_extension/dataset_manifest.json","Official dataset provenance"),("results/run_metadata/synthetic_run.json","Full synthetic execution and checkpoint manifest")]:
        path=root/relative
        if path.exists():
            content+=f'<details><summary>{esc(title)}</summary><pre class="plain-source"><code>{esc(path.read_text())}</code></pre></details>'
    backup=root/'backups/baseline_1.py'
    content+='<details><summary>Read the complete preserved original implementation</summary>'+highlight(backup.read_text(),PythonLexer(),HtmlFormatter(cssclass="code-highlight",linenos="table"))+'</details>'
    content+='<p class="note">An unchanged source hash demonstrates byte identity; passing tests demonstrate the behaviors those tests exercised. Neither alone proves hardware fidelity, a global capture theorem, arbitrary long-horizon rare-event accuracy or validity of an LLM phase representation.</p>'
    return content


CSS = r'''
:root{color-scheme:light;--paper:#f7f5ed;--panel:#efeee6;--ink:#1d302f;--muted:#536461;--line:#c8cebf;--accent:#006b62;--accent-ink:#004c45;--code:#eeeee6;--focus:#b24c21;--rail:278px}
:root[data-theme="dark"]{color-scheme:dark;--paper:#172321;--panel:#202e2b;--ink:#eef0e6;--muted:#bac9c0;--line:#53615b;--accent:#83d5c3;--accent-ink:#a1e8d8;--code:#1e2d29;--focus:#ffb58b}
*{box-sizing:border-box}html{scroll-behavior:auto;scroll-padding-top:28px}body{margin:0;background:var(--paper);color:var(--ink);font:19px/1.7 Georgia,"Times New Roman",serif}a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{text-decoration-thickness:2px}button,input,select,summary{font-family:Arial,Helvetica,sans-serif}button,input,select{font-size:14px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:3px}button,select{min-height:42px;padding:9px 12px;cursor:pointer}button:hover{border-color:var(--accent)}input[type="search"]{width:100%;min-height:44px;padding:10px 11px}input[type="range"]{width:100%;accent-color:var(--accent)}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,a:focus-visible,.table-scroll:focus-visible{outline:3px solid var(--focus);outline-offset:3px}::selection{background:#a9d8c9;color:#102c26}.skip{position:fixed;left:10px;top:-100px;padding:12px;background:var(--paper);z-index:50}.skip:focus{top:10px}
.rail{position:fixed;inset:0 auto 0 0;width:var(--rail);padding:36px 26px 25px;border-right:1px solid var(--line);overflow-y:auto;font:14px/1.5 Arial,Helvetica,sans-serif}.rail-brand{margin:0 0 20px;color:var(--ink);font:24px/1.15 Georgia,serif}.rail-brand span{display:block;margin-top:9px;font:12px/1.5 Arial,sans-serif;color:var(--muted)}.rail label{display:block;margin:15px 0 7px;font-weight:bold}.search-status{font-size:12px;min-height:20px;color:var(--muted);margin:7px 0 15px}.rail nav a{display:flex;gap:10px;margin:3px -8px;padding:8px;text-decoration:none;border-left:3px solid transparent;color:var(--ink)}.rail nav a .letter{width:18px;color:var(--accent);font-weight:bold}.rail nav a.active{border-left-color:var(--accent);background:var(--panel)}.rail nav a[hidden]{display:none}.rail-controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.rail-controls button{font-size:12px;padding:8px}.rail-foot{border-top:1px solid var(--line);margin-top:24px;padding-top:16px;font-size:12px;color:var(--muted)}.mobile-bar{display:none}
main{margin-left:var(--rail);padding:65px clamp(28px,5vw,100px) 90px;max-width:1550px}.masthead{border-bottom:2px solid var(--ink);padding-bottom:44px;margin-bottom:58px;max-width:1080px}.eyebrow{font:13px/1.4 Arial,sans-serif;color:var(--muted);margin:0 0 21px}.masthead h1{font-size:clamp(42px,5.1vw,76px);line-height:1.03;letter-spacing:-1.5px;font-weight:normal;margin:0 0 25px;max-width:880px}.masthead .subtitle{max-width:750px;font-size:23px;line-height:1.5;margin:0}.masthead .meta{font:13px/1.65 Arial,sans-serif;margin-top:27px;color:var(--muted)}section{border-bottom:1px solid var(--line);padding-bottom:65px;margin-bottom:65px;min-width:0;max-width:1120px}section[hidden]{display:none}.section-head{display:flex;align-items:baseline;gap:20px;margin-bottom:30px}.section-letter{font:36px/1 Georgia,serif;color:var(--accent)}h2{font-size:38px;line-height:1.18;font-weight:normal;margin:0}h3{font-size:27px;line-height:1.3;font-weight:normal;margin:42px 0 16px}h4{font:600 18px/1.5 Arial,sans-serif;margin:25px 0 14px}h5{font:600 15px/1.5 Arial,sans-serif;margin:22px 0 12px}p,li{max-width:83ch}p{margin:0 0 20px}li{padding-left:4px;margin:7px 0}ol,ul{padding-left:25px}strong{font-weight:bold}.lede{font-size:26px;line-height:1.5;max-width:760px;margin-bottom:28px}.note{font:15px/1.65 Arial,sans-serif;color:var(--muted);border-left:3px solid var(--accent);padding:12px 0 12px 18px;margin:24px 0}.implementation{font:16px/1.7 Arial,sans-serif;color:var(--muted)}.equation{margin:32px 0 40px;padding:9px 0 0;border-top:1px solid var(--line)}.math-display{width:100%;overflow-x:auto;overflow-y:hidden;padding:18px 10px;margin:14px 0 20px;font-size:1.05em}.math-display mjx-container{max-width:none!important}mjx-container[display="true"]{margin:.6em 0!important}mjx-container svg{max-width:none!important}.math-code-map{margin:30px 0}.module{margin:34px 0 50px}.module h3{font-size:22px;margin-bottom:20px}.module h3 span{display:block;font:24px/1.4 Georgia,serif;margin-top:12px}.experiment{border-top:1px solid var(--line);padding-top:1px;margin-top:25px}.source-ref{font-size:12px;color:var(--muted)}code{font:0.83em/1.5 "DejaVu Sans Mono",Consolas,monospace;overflow-wrap:anywhere}.table-scroll{overflow-x:auto;margin:22px 0 28px;max-width:100%;border-block:1px solid var(--line)}table{width:100%;border-collapse:collapse;text-align:left;font:13px/1.65 Arial,Helvetica,sans-serif}caption{text-align:left;font:13px/1.6 Arial,sans-serif;color:var(--muted);padding:12px 4px;caption-side:bottom}th,td{padding:12px 11px;vertical-align:top;border-bottom:1px solid var(--line);min-width:90px}th{font-weight:bold;color:var(--ink);background:var(--panel)}tbody tr:last-child td{border-bottom:0}td:first-child{font-weight:500}td code{white-space:pre-wrap}.equation table td:first-child{width:25%;min-width:135px}.equation table td{font-size:14px}.table-scroll table{min-width:600px}.equation .table-scroll table{min-width:440px}details{margin:16px 0;border:1px solid var(--line);border-radius:3px;min-width:0}summary{padding:14px 17px;cursor:pointer;font-size:14px;line-height:1.6;font-weight:600;overflow-wrap:anywhere}summary:hover{background:var(--panel)}details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:17px}details>*:not(summary){margin-left:18px;margin-right:18px}details>p:last-child{margin-bottom:24px}details .table-scroll{margin-left:0;margin-right:0}.function summary span{display:block;font-size:12px;font-weight:normal;color:var(--muted);margin-left:16px}.function .execution{font-size:16px}.execution code{font-size:12px}.line-ref{font:11px Arial,sans-serif;color:var(--accent);display:inline-block;min-width:46px}blockquote{border-left:2px solid var(--line);padding-left:15px;color:var(--muted);font-size:16px}pre{font:12px/1.65 "DejaVu Sans Mono",Consolas,monospace;tab-size:4}.plain-source{padding:18px;background:var(--code);overflow:auto;white-space:pre;max-height:650px}.plain-source code{font:inherit;overflow-wrap:normal}.code-highlight{max-width:100%;overflow:auto;background:var(--code);font-size:12px}.code-highlight table{width:auto;min-width:100%;border:0;background:var(--code)}.code-highlight td{padding:10px 8px;border:0;min-width:0}.code-highlight pre{margin:0;white-space:pre;overflow-wrap:normal;color:var(--ink)}.code-highlight .linenos{user-select:none;color:var(--muted);border-right:1px solid var(--line);text-align:right;width:36px}.code-highlight .linenos pre{color:var(--muted)}.code-highlight .k,.code-highlight .kn,.code-highlight .kd,.code-highlight .ow{color:#075d70;font-weight:bold}.code-highlight .s,.code-highlight .s1,.code-highlight .s2,.code-highlight .sd{color:#86520b}.code-highlight .c,.code-highlight .c1,.code-highlight .cm{color:#566c54;font-style:italic}.code-highlight .nf,.code-highlight .nc{color:#725090}.code-highlight .mi,.code-highlight .mf{color:#8c3e44}:root[data-theme="dark"] .code-highlight .k,:root[data-theme="dark"] .code-highlight .kn,:root[data-theme="dark"] .code-highlight .kd,:root[data-theme="dark"] .code-highlight .ow{color:#82d7db}:root[data-theme="dark"] .code-highlight .s,:root[data-theme="dark"] .code-highlight .s1,:root[data-theme="dark"] .code-highlight .s2,:root[data-theme="dark"] .code-highlight .sd{color:#efca8f}:root[data-theme="dark"] .code-highlight .c,:root[data-theme="dark"] .code-highlight .c1,:root[data-theme="dark"] .code-highlight .cm{color:#b3c4a6}:root[data-theme="dark"] .code-highlight .nf,:root[data-theme="dark"] .code-highlight .nc{color:#ceb0e8}:root[data-theme="dark"] .code-highlight .mi,:root[data-theme="dark"] .code-highlight .mf{color:#f4b8bd}
figure{margin:34px 0 40px}figcaption{font:14px/1.75 Arial,Helvetica,sans-serif;margin:14px 0 0;max-width:100ch}figcaption span{display:block;margin-top:8px}.figure-kicker{font:12px/1.5 Arial,sans-serif;color:var(--muted);padding:10px 0;border-top:1px solid var(--line)}.concept-svg{width:100%;height:auto;display:block;min-width:0}.concept{padding:6px 0 20px}.measured img{display:block;width:100%;height:auto;background:white;border:1px solid var(--line)}.source-page{display:block;max-width:100%;height:auto;margin:auto}.source-pages figure{margin:24px auto}.interactive{margin:34px 0 46px;border-top:2px solid var(--accent);padding-top:1px}.controls{font:14px/1.5 Arial,sans-serif;display:flex;flex-wrap:wrap;gap:18px 30px;margin:22px 0}.controls label{display:flex;flex-direction:column;gap:8px;flex:1;min-width:min(260px,100%)}.controls select{width:100%;max-width:100%}#baseline-interactive{width:100%;height:auto;display:block;border-block:1px solid var(--line)}output{display:block;font:14px/1.8 Arial,sans-serif;padding:15px 0;color:var(--ink);min-height:65px}.grid-scroll{overflow:auto;max-width:100%}.grid-scroll table{min-width:680px}.grid-scroll td,.grid-scroll th{padding:5px;text-align:center;min-width:70px}.grid-cell{display:block;width:100%;border:1px solid transparent;min-height:46px;border-radius:1px;font-size:12px;padding:7px 4px}.grid-cell[aria-pressed="true"]{outline:3px solid var(--focus);outline-offset:-3px}.grid-axis{font:13px Arial,sans-serif;margin:12px 0 8px;color:var(--muted)}.download{margin:0 0 20px;font-size:12px}.empty-state{padding:35px 0;margin-bottom:40px;max-width:650px}.empty-state[hidden]{display:none}.empty-state h2{font-size:28px}.empty-state p{margin-top:15px}footer{font:13px/1.8 Arial,sans-serif;color:var(--muted);max-width:100ch;border-top:2px solid var(--ink);padding-top:25px;margin-top:30px}.noscript{border:1px solid var(--line);padding:15px;font:14px Arial,sans-serif}
@media(min-width:1450px){main{padding-left:90px;padding-right:95px}.rail{width:290px}:root{--rail:290px}}
@media(max-width:1050px){:root{--rail:235px}.rail{padding:28px 18px}.rail-brand{font-size:21px}main{padding:45px 28px}.masthead h1{font-size:56px}body{font-size:18px}h2{font-size:33px}h3{font-size:25px}}
@media(max-width:760px){:root{--rail:0px}.mobile-bar{display:flex;position:sticky;top:0;z-index:20;align-items:center;justify-content:space-between;gap:10px;padding:10px 16px;background:var(--paper);border-bottom:1px solid var(--line);font:15px/1.3 Arial,sans-serif}.rail{display:none;position:fixed;top:64px;width:min(340px,100%);z-index:30;background:var(--paper);box-shadow:10px 0 24px #0002}.rail.open{display:block}main{margin:0;padding:30px 20px 60px}.masthead{padding-bottom:32px;margin-bottom:35px}.masthead h1{font-size:42px;letter-spacing:-.7px}.masthead .subtitle{font-size:21px}.masthead .meta{font-size:12px}.eyebrow{font-size:12px}.section-head{gap:14px}.section-letter{font-size:29px}h2{font-size:29px}.lede{font-size:22px}h3{font-size:24px;margin-top:32px}body{font-size:17px;line-height:1.7}section{margin-bottom:42px;padding-bottom:42px}.equation{margin:25px 0}.math-display{font-size:.9em;padding:15px 0}.concept-svg{min-width:620px}.concept{overflow-x:auto}.concept figcaption{min-width:0;max-width:100%;position:sticky;left:0}.table-scroll table{min-width:600px}.equation .table-scroll table{min-width:400px}th,td{padding:10px 8px}.controls{display:block}.controls label{margin:15px 0}.controls select{font-size:13px}.code-highlight{font-size:11px}details>*:not(summary){margin-left:12px;margin-right:12px}.module h3 span{font-size:22px}.source-ref code{word-break:break-all}.rail nav a{min-height:42px}summary{font-size:13px;padding:13px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
@media print{.rail,.mobile-bar,.controls,button,.skip{display:none!important}main{margin:0;padding:0;max-width:none}body{background:white;color:black;font-size:11pt}section{page-break-before:always;max-width:none}section[hidden]{display:block!important}.masthead{page-break-after:always}details>summary{font-size:10pt}.measured,.concept,.equation{break-inside:avoid}a{color:inherit}.table-scroll{overflow:visible}table{font-size:8pt}.source-ref{font-size:7pt}}
'''


JAVASCRIPT = r'''
"use strict";
(() => {
const bodyData = JSON.parse(document.getElementById("report-data").textContent);
const sections = [...document.querySelectorAll("main section")];
const links = [...document.querySelectorAll(".rail nav a")];
const search = document.getElementById("section-search");
const status = document.getElementById("search-status");
const empty = document.getElementById("no-results");
const rail = document.getElementById("contents");
const menu = document.getElementById("contents-toggle");
const textIndex = sections.map(section => section.textContent.toLowerCase());
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = document.getElementById("theme-toggle");
  button.textContent = theme === "dark" ? "Light theme" : "Dark theme";
  button.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " theme");
  try { localStorage.setItem("pll-report-theme",theme); } catch (_) {}
  drawBaseline();
}
let initialTheme="light";
try { initialTheme=localStorage.getItem("pll-report-theme") || "light"; } catch (_) {}
document.getElementById("theme-toggle").addEventListener("click",() => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
function filterSections() {
  const query = search.value.trim().toLowerCase();
  const terms = query.split(/\s+/).filter(Boolean);
  let count = 0;
  sections.forEach((section,i) => {
    const match = terms.every(term => textIndex[i].includes(term));
    section.hidden = !match;
    links[i].hidden = !match;
    if (match) count++;
  });
  status.textContent = query ? `${count} of ${sections.length} sections match` : "All 10 sections available";
  empty.hidden = count !== 0;
  if (!count) document.getElementById("empty-query").textContent=search.value;
}
search.addEventListener("input",filterSections);
document.getElementById("reset-search").addEventListener("click",() => {search.value="";filterSections();search.focus();});
document.getElementById("clear-search").addEventListener("click",() => {search.value="";filterSections();search.focus();});
menu.addEventListener("click",() => {const open=rail.classList.toggle("open");menu.setAttribute("aria-expanded",String(open));});
document.addEventListener("keydown",event => {if(event.key==="Escape"){rail.classList.remove("open");menu.setAttribute("aria-expanded","false");}});
links.forEach(link => link.addEventListener("click",() => {rail.classList.remove("open");menu.setAttribute("aria-expanded","false");}));
document.getElementById("expand-details").addEventListener("click",() => document.querySelectorAll("main section:not([hidden]) details").forEach(detail => detail.open=true));
document.getElementById("collapse-details").addEventListener("click",() => document.querySelectorAll("main details").forEach(detail => detail.open=false));
if ("IntersectionObserver" in window) {
 const observer=new IntersectionObserver(entries => entries.forEach(entry => {if(entry.isIntersecting){links.forEach(link => link.classList.toggle("active",link.hash==="#"+entry.target.id));}}),{rootMargin:"-10% 0px -70% 0px"});
 sections.forEach(section => observer.observe(section));
}
document.querySelectorAll("[data-download]").forEach(button => button.addEventListener("click",() => {
 const name=button.dataset.download, rows=bodyData.tables[name];
 if(!rows || !rows.length) return;
 const fields=Object.keys(rows[0]);
 const quote=value => '"'+String(value==null?"":value).replace(/"/g,'""')+'"';
 const csv=[fields.map(quote).join(","),...rows.map(row=>fields.map(field=>quote(row[field])).join(","))].join("\r\n");
 const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
 const anchor=document.createElement("a");anchor.href=url;anchor.download=name+".csv";anchor.click();setTimeout(()=>URL.revokeObjectURL(url),500);
}));
const NS="http://www.w3.org/2000/svg";
function svgElement(tag,attributes,text) {
 const element=document.createElementNS(NS,tag);
 Object.entries(attributes||{}).forEach(([key,value])=>element.setAttribute(key,String(value)));
 if(text!=null) element.textContent=text;
 return element;
}
const series=bodyData.series;
const slider=document.getElementById("baseline-time");slider.max=series.t.length-1;slider.value=series.t.length-1;
function fmt(value) {return value==null || !Number.isFinite(Number(value)) ? "not applicable" : Number(value).toPrecision(4).replace(/\.0+(?=e|$)/,"");}
function drawBaseline() {
 const chart=document.getElementById("baseline-interactive");if(!chart)return;
 chart.replaceChildren();
 const view=document.getElementById("baseline-view").value;
 const dark=document.documentElement.dataset.theme==="dark";
 const ink=dark?"#eef0e6":"#1d302f",muted=dark?"#bac9c0":"#536461",teal=dark?"#83d5c3":"#006b62";
 const color2=dark?"#efbf91":"#a25821";
 const selected=view==="variance" ? ["variance_phi"] : view==="mean" ? ["mean_phi"] : ["deterministic_phi","stochastic_phi","mean_phi"];
 const all=selected.flatMap(key=>series[key]).filter(Number.isFinite);
 if(view==="mean") all.push(...series.mean_ci_low.filter(Number.isFinite),...series.mean_ci_high.filter(Number.isFinite));
 let ymin=Math.min(...all),ymax=Math.max(...all);const padding=Math.max((ymax-ymin)*.12,.00001);ymin-=padding;ymax+=padding;
 const left=90,right=965,top=35,bottom=315,tmax=series.t[series.t.length-1];
 const x=time=>left+(right-left)*time/tmax,y=value=>bottom-(bottom-top)*(value-ymin)/(ymax-ymin);
 for(let tick=0;tick<=5;tick++){
   const tx=left+(right-left)*tick/5,vy=ymin+(ymax-ymin)*tick/5,py=y(vy);
   chart.append(svgElement("line",{x1:tx,y1:top,x2:tx,y2:bottom,stroke:muted,"stroke-opacity":.14}));
   chart.append(svgElement("line",{x1:left,y1:py,x2:right,y2:py,stroke:muted,"stroke-opacity":.14}));
   chart.append(svgElement("text",{x:tx,y:340,"text-anchor":"middle",fill:ink,"font-size":13,"font-family":"Arial"},fmt(tmax*tick/5)));
   chart.append(svgElement("text",{x:left-12,y:py+5,"text-anchor":"end",fill:ink,"font-size":12,"font-family":"Arial"},fmt(vy)));
 }
 chart.append(svgElement("path",{d:`M${left},${top}V${bottom}H${right}`,fill:"none",stroke:ink}));
 chart.append(svgElement("text",{x:520,y:372,"text-anchor":"middle",fill:ink,"font-size":15,"font-family":"Arial"},"Time (model units)"));
 chart.append(svgElement("text",{x:17,y:175,transform:"rotate(-90 17 175)","text-anchor":"middle",fill:ink,"font-size":14,"font-family":"Arial"},view==="variance"?"Phase variance (rad²)":"Phase (rad)"));
 if(view==="mean"){
   const high=series.mean_ci_high.map((value,i)=>`${x(series.t[i])},${y(value)}`),low=series.mean_ci_low.map((value,i)=>`${x(series.t[i])},${y(value)}`).reverse();
   chart.append(svgElement("polygon",{points:[...high,...low].join(" "),fill:teal,"fill-opacity":.18,stroke:"none"}));
 }
 const colors=view==="compare"?[ink,color2,teal]:[teal];
 const names={deterministic_phi:"Deterministic φ",stochastic_phi:"One stochastic φ",mean_phi:"Ensemble mean φ",variance_phi:"Across-path variance"};
 selected.forEach((key,j)=>{
   chart.append(svgElement("polyline",{points:series[key].map((value,i)=>`${x(series.t[i])},${y(value)}`).join(" "),fill:"none",stroke:colors[j],"stroke-width":key==="mean_phi"?2.6:1.5}));
   const lx=100+j*270;chart.append(svgElement("line",{x1:lx,y1:398,x2:lx+26,y2:398,stroke:colors[j],"stroke-width":3}));
   chart.append(svgElement("text",{x:lx+35,y:403,fill:ink,"font-size":13,"font-family":"Arial"},names[key]));
 });
 if(view==="mean")chart.append(svgElement("text",{x:430,y:403,fill:ink,"font-size":13,"font-family":"Arial"},"Shading: pointwise 95% CI for the mean"));
 const index=Number(slider.value),time=series.t[index];
 chart.append(svgElement("line",{x1:x(time),y1:top,x2:x(time),y2:bottom,stroke:muted,"stroke-dasharray":"4 4"}));
 selected.forEach((key,j)=>chart.append(svgElement("circle",{cx:x(time),cy:y(series[key][index]),r:4,fill:colors[j]})));
 document.getElementById("baseline-readout").textContent=`t=${fmt(time)}: deterministic φ=${fmt(series.deterministic_phi[index])} rad; stochastic φ=${fmt(series.stochastic_phi[index])} rad; mean φ=${fmt(series.mean_phi[index])} rad, 95% mean CI [${fmt(series.mean_ci_low[index])}, ${fmt(series.mean_ci_high[index])}]; phase variance=${fmt(series.variance_phi[index])} rad².`;
 slider.setAttribute("aria-valuetext",fmt(time)+" model time units");
}
document.getElementById("baseline-view").addEventListener("change",drawBaseline);slider.addEventListener("input",drawBaseline);
function drawGrid(){
 const field=document.getElementById("grid-metric").value,rows=bodyData.tables.combined_summary;
 const deltas=[...new Set(rows.map(row=>row.delta))].sort((a,b)=>a-b),alphas=[...new Set(rows.map(row=>row.alpha))].sort((a,b)=>a-b);
 const values=rows.map(row=>row[field]).filter(value=>value!=null&&Number.isFinite(value));
 const lo=Math.min(...values),hi=Math.max(...values);const table=document.createElement("table");
 const caption=document.createElement("caption");caption.textContent=`Measured cells; color spans ${fmt(lo)} to ${fmt(hi)} for the selected quantity. Rows: normalized detuning δ; columns: noise amplitude α.`;table.append(caption);
 const thead=document.createElement("thead"),header=document.createElement("tr");
 const corner=document.createElement("th");corner.textContent="δ \\ α";corner.scope="col";header.append(corner);
 alphas.forEach(alpha=>{const th=document.createElement("th");th.textContent=alpha;th.scope="col";header.append(th);});thead.append(header);table.append(thead);
 const tbody=document.createElement("tbody");
 deltas.forEach(delta=>{
   const tr=document.createElement("tr"),th=document.createElement("th");th.textContent=delta;th.scope="row";tr.append(th);
   alphas.forEach(alpha=>{
     const row=rows.find(row=>row.delta===delta&&row.alpha===alpha),td=document.createElement("td"),button=document.createElement("button"),value=row[field];
     button.type="button";button.className="grid-cell";button.textContent=value==null?"n/a":fmt(value);button.setAttribute("aria-pressed","false");
     const norm=value==null?0:(value-lo)/(hi-lo||1),r=Math.round(232-222*norm),g=Math.round(241-134*norm),b=Math.round(235-135*norm);
     if(value!=null){
       button.style.backgroundColor=`rgb(${r},${g},${b})`;
       const linear=channel=>{const s=channel/255;return s<=.04045?s/12.92:((s+.055)/1.055)**2.4;};
       const luminance=.2126*linear(r)+.7152*linear(g)+.0722*linear(b);
       button.style.color=luminance>.179?"#000000":"#ffffff";
     }
     button.setAttribute("aria-label",`δ ${delta}, α ${alpha}, n ${row.n}, ${field.replaceAll("_"," ")} ${value==null?"not applicable":fmt(value)}`);
     function inspect(){
       document.querySelectorAll(".grid-cell").forEach(cell=>cell.setAttribute("aria-pressed","false"));button.setAttribute("aria-pressed","true");
       const lower=row[field+"_low"],upper=row[field+"_high"],ci=lower!=null&&upper!=null?`95% interval [${fmt(lower)}, ${fmt(upper)}]`:alpha===0?"single deterministic path; no Monte Carlo interval":"no interval stored for this quantity";
       document.getElementById("grid-readout").textContent=`Condition ${row.condition_id}: δ=${delta}, α=${alpha}, n=${row.n}, T=${row.T}, dt=${row.dt}. ${field.replaceAll("_"," ")}=${fmt(value)}; ${value==null?"locked-well observable is not applicable":ci}.`;
     }
     button.addEventListener("click",inspect);button.addEventListener("focus",inspect);td.append(button);tr.append(td);
   });tbody.append(tr);
 });table.append(tbody);document.getElementById("grid-interactive").replaceChildren(table);
}
document.getElementById("grid-metric").addEventListener("change",drawGrid);
setTheme(initialTheme);drawGrid();filterSections();
})();
'''


def document_shell(content, data, vendor, generated):
    """Assemble the responsive offline document with no external runtime references."""
    nav="".join(f'<a href="#{identifier}"><span class="letter">{letter}</span><span>{title}</span></a>' for identifier,letter,title in SECTIONS)
    math_config=r'''window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']],packages:['base','ams','newcommand','noundefined']},svg:{fontCache:'local'},options:{enableMenu:false,enableAssistiveMml:false},startup:{typeset:true}};'''
    # The configuration string above is authored as JavaScript, so one literal
    # backslash must reach the MathJax delimiter rather than two.
    math_config=math_config.replace("\\\\\\\\", "\\\\")
    return ('<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        '<meta name="description" content="A complete offline teaching and research report on stochastic phase-locked loops: source-verified mathematics, full local experiments, independent-path uncertainty, code and reproducibility."/>'
        '<meta name="color-scheme" content="light dark"/><title>Stochastic phase-locked loops | Complete research report</title>'
        '<style>'+CSS+'</style></head><body><a class="skip" href="#main-content">Skip to report content</a>'
        '<div class="mobile-bar"><span>Stochastic PLL research</span><button id="contents-toggle" type="button" aria-controls="contents" aria-expanded="false">Contents</button></div>'
        '<aside class="rail" id="contents" aria-label="Report contents"><p class="rail-brand">Stochastic<br/>phase-locked loops<span>A complete mathematical and computational study</span></p>'
        '<label for="section-search">Search the report sections</label><input id="section-search" type="search" placeholder="e.g. Brownian, censoring" autocomplete="off"/>'
        '<p class="search-status" id="search-status" role="status">All 10 sections available</p><nav aria-label="Sections A through J">'+nav+'</nav>'
        '<div class="rail-controls"><button type="button" id="theme-toggle">Dark theme</button><button type="button" id="clear-search">Clear search</button><button type="button" id="expand-details">Expand explanations</button><button type="button" id="collapse-details">Collapse explanations</button></div>'
        '<div class="rail-foot">Source pages 273–275<br/>Kloeden and Platen, Chapter 7<br/>Local full-scale experiments<br/>Read offline, directly from this file</div></aside>'
        '<main id="main-content"><header class="masthead"><p class="eyebrow">University research project · stochastic numerical analysis · communication systems</p>'
        '<h1>When a phase lock<br/>meets noise.</h1><p class="subtitle">From the textbook equation to completed transitions, full-scale experiments and the limits of synchronization analogies.</p>'
        f'<p class="meta">Complete teaching and research report · Generated {esc(generated)}<br/>Phase in radians; time in model units · All measured findings come from saved local outputs.</p></header>'
        '<noscript><p class="noscript">All sections, equations as TeX, tables, figures and code remain readable. JavaScript enables equation typesetting, search, themes, downloads and the two actual-data plot controls.</p></noscript>'
        '<div class="empty-state" id="no-results" hidden><h2>No matching sections</h2><p>No section contains every word in “<span id="empty-query"></span>”. Try a shorter term such as <em>phase</em>, <em>variance</em> or <em>seed</em>.</p><button id="reset-search" type="button">Show all sections</button></div>'
        +content+'<footer>This document separates source-verified mathematics, executed numerical evidence and proposed extensions. Scientific figures, source-page images, current code, summary data, styles and MathJax SVG rendering are embedded. External reference links are optional reading and are not required for offline use.<br/>Design: ENERGY 1 / RHYTHM 2 / MOTION 1. MathJax 3.2.2 is distributed under the Apache 2.0 license; the license is retained with the project vendor copy.</footer></main>'
        '<script type="application/json" id="report-data">'+safe_json(data)+'</script>'
        '<script>'+JAVASCRIPT.replace('</script','<\\/script')+'</script>'
        '<script>'+math_config+'</script><script>'+vendor.replace('</script','<\\/script')+'</script></body></html>')


def generate(root=ROOT, output=None):
    """Validate evidence, compose A–J, embed assets, and atomically write the final report."""
    root=Path(root).resolve()
    metadata,tables,series,figures,captions=load_inputs(root)
    output=Path(output) if output else root/'report/stochastic_pll_complete_report.html'
    generated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    displayed=datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %B %Y, %I:%M %p IST')
    bodies={"overview":overview(metadata,tables),"foundations":foundations(root),"visual":
            '<p>These thirteen diagrams are conceptual explanations. Their curves and arrows illustrate mechanisms; numerical results are separately labeled in Section F. Read each diagram with its caption to understand what is represented and what is only schematic.</p>'+diagrams(),
            "code":code_walkthrough(root),"methods":methods(root,tables),
            "results":results(root,metadata,tables,figures,captions),"literature":literature(root),
            "discussion":discussion(tables),"conclusion":conclusion(metadata,tables),
            "reproducibility":reproducibility(root,metadata,tables,figures)}
    content="".join(f'<section id="{identifier}" aria-labelledby="heading-{identifier}"><div class="section-head"><span class="section-letter" aria-hidden="true">{letter}</span><h2 id="heading-{identifier}">{title}</h2></div>{bodies[identifier]}</section>' for identifier,letter,title in SECTIONS)
    data={"metadata":metadata,"tables":tables,"series":series}
    vendor=(root/'report/vendor/mathjax-3.2.2-tex-svg.js').read_text()
    document=document_shell(content,data,vendor,displayed)
    for pattern in [r'<script[^>]+src\s*=',r'<link[^>]+href\s*=',r'<img[^>]+src\s*=\s*["\'](?!data:)']:
        if re.search(pattern,document,re.I):
            raise ValueError(f"Report contains a nonembedded runtime/image dependency: {pattern}")
    if document.count('class="concept"')!=13:
        raise ValueError("The thirteen explanatory SVG diagrams are incomplete")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp');temporary.write_text(document,encoding='utf-8');temporary.replace(output)
    embedded=figures+[root/f'notes/textbook_p{page}.png' for page in [273,274,275]]+[root/'report/vendor/mathjax-3.2.2-tex-svg.js']
    manifest={"generated_utc":generated,"report":str(output.relative_to(root)),"report_bytes":output.stat().st_size,
              "report_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"conceptual_diagrams":13,
              "measured_or_analytical_saved_figures":len(figures),"summary_tables":{name:len(rows) for name,rows in tables.items()},
              "runtime_external_references":0,"browser_verified":False,
              "assets":[{"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in embedded]}
    (root/'report/report_assets.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    """Run the report builder from a project root after all experiment artifacts exist."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    arguments=parser.parse_args()
    print(json.dumps(generate(arguments.project,arguments.output),indent=2))


if __name__=='__main__':
    main()
