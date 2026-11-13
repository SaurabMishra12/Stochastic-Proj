"""Build the concise course report from saved, completed experiment outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text())


def fmt(value, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return r"\textemdash"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10000:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"
    return f"{value:.{digits}g}"


def ci(row: dict, key: str, digits: int = 3) -> str:
    return (
        fmt(row[key], digits)
        + " ["
        + fmt(row.get(key + "_low"), digits)
        + ", "
        + fmt(row.get(key + "_high"), digits)
        + "]"
    )


def escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def make_table(headers, rows, spec: str, caption: str, label: str) -> str:
    content = [
        r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + spec + r"}\toprule",
        " & ".join(headers) + r" \\ \midrule",
    ]
    content.extend(" & ".join(str(value) for value in row) + r" \\" for row in rows)
    content.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\end{table}",
        ]
    )
    return "\n".join(content) + "\n"


def figure(name: str, caption: str, label: str, height: str = ".38") -> str:
    return (
        r"\begin{figure}[htbp]\centering" + "\n"
        + r"\includegraphics[width=.96\linewidth,height=" + height
        + r"\textheight,keepaspectratio]{figures/" + name + ".pdf}\n"
        + r"\caption{" + caption + "}\n"
        + r"\label{" + label + "}\n"
        + r"\end{figure}" + "\n"
    )


def load_conditions():
    rows = [read_json(path) for path in sorted((ROOT / "results/raw_data").glob("condition_*/summary.json"))]
    by_id = {row["condition_id"]: row for row in rows}
    families = {
        "grid": [by_id[i] for i in range(100, 172) if i in by_id],
        "acquisition": [by_id[i] for i in [200] if i in by_id],
        "long": [by_id[i] for i in range(300, 306) if i in by_id],
        "sensitivity": [by_id[i] for i in range(400, 418) if i in by_id],
    }
    for key, expected in [("grid", 72), ("acquisition", 1), ("long", 6), ("sensitivity", 18)]:
        if len(families[key]) != expected:
            raise RuntimeError(f"Cannot generate completed report: {key} has {len(families[key])}/{expected} summaries")
    for row in rows:
        if not math.isfinite(row["phi_variance"]) or row["phi_variance"] < 0:
            raise ValueError(f"Invalid saved phase variance: {row['condition_id']}")
    return families


def find_cell(rows, delta, alpha):
    matches = [row for row in rows if math.isclose(row["delta"], delta) and math.isclose(row["alpha"], alpha)]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell at delta={delta}, alpha={alpha}; found {len(matches)}")
    return matches[0]


def convergence_rows():
    path = ROOT / "results/tables/convergence_summary.csv"
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: value if key == "scenario" else float(value) for key, value in row.items()})
    if len(rows) != 10 or {row["scenario"] for row in rows} != {"baseline", "near_threshold"}:
        raise RuntimeError("Both completed five-mesh convergence scenarios are required")
    if any(row["finite_fraction"] != 1 for row in rows):
        raise RuntimeError("A convergence scenario contains nonfinite state values")
    return rows


def embedded_bibliography() -> str:
    text = (ROOT / "report/references.bib").read_text()
    entries = []
    position = 0
    while (start := text.find("@", position)) != -1:
        opening = text.index("{", start)
        kind = text[start + 1:opening].strip().lower()
        comma = text.index(",", opening)
        key = text[opening + 1:comma].strip()
        fields = {}
        cursor = comma + 1
        while cursor < len(text):
            while text[cursor].isspace() or text[cursor] == ",":
                cursor += 1
            if text[cursor] == "}":
                position = cursor + 1
                break
            equal = text.index("=", cursor)
            name = text[cursor:equal].strip().lower()
            cursor = equal + 1
            while text[cursor].isspace():
                cursor += 1
            if text[cursor] != "{":
                raise ValueError(f"Expected brace-delimited bibliography field in {key}:{name}")
            depth, value_start = 1, cursor + 1
            cursor += 1
            while depth:
                depth += (text[cursor] == "{") - (text[cursor] == "}")
                cursor += 1
            fields[name] = text[value_start:cursor - 1]
        entries.append((kind, key, fields))
    expected = {"kloeden1992", "higham2001", "bobrovsky1982", "bianchi2015", "kramers1940", "khasminskii2012", "acebron2005", "li2024", "cobbe2021", "gsm8krepository", "wang2026"}
    if {key for _, key, _ in entries} != expected or len(entries) != len(expected):
        raise ValueError("Reusable bibliography must contain exactly the eleven verified citation records")
    lines = [r"\begin{thebibliography}{11}"]
    for kind, key, fields in entries:
        author = fields["author"].replace(" and ", "; ").rstrip(".")
        title = fields["title"]
        if kind == "book":
            body = rf"{author}. \emph{{{title}}}. {fields['publisher']}, "
            if "edition" in fields:
                body += fields["edition"] + "nd edition, "
            body += fields["year"] + "."
        elif kind == "article":
            body = rf"{author}. {title}. \emph{{{fields['journal']}}}"
            if "volume" in fields:
                body += rf", \textbf{{{fields['volume']}}}"
                if "number" in fields:
                    body += "(" + fields["number"] + ")"
                body += ", pp.~" + fields["pages"]
            body += " (" + fields["year"] + ")."
        elif kind == "inproceedings":
            body = rf"{author}. {title}. In \emph{{{fields['booktitle']}}}, pp.~{fields['pages']} ({fields['year']})."
        else:
            body = rf"{author}. \emph{{{title}}}. {fields['howpublished']}. Accessed 13 November 2026."
        if "doi" in fields:
            body += r" \url{https://doi.org/" + fields["doi"] + "}."
        else:
            body += r" \url{" + fields["url"] + "}."
        lines.append(r"\bibitem{" + key + "} " + body)
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines) + "\n"


def synthetic_results() -> tuple[str, int, int]:
    path = ROOT / "results/tables/synthetic_summary.csv"
    if not path.exists():
        return "", 0, 0
    with path.open(newline="") as handle:
        rows = [{key: value if key == "topology" else float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != 144 or any(row["n"] != 500 for row in rows):
        raise RuntimeError("Synthetic report requires all 144 completed 500-network conditions")
    def cell(n_agents, topology, rho, dropout=0, delay=0):
        selected = [row for row in rows if row["n_agents"] == n_agents and row["topology"] == topology and row["rho"] == rho and row["dropout"] == dropout and row["delay"] == delay]
        if len(selected) != 1:
            raise ValueError(f"Missing or duplicate synthetic condition: {n_agents}, {topology}, {rho}, {dropout}, {delay}")
        return selected[0]
    chosen = [cell(n, topology, 0) for n in [8, 20] for topology in ["complete", "ring", "star", "sparse"]]
    chosen += [cell(3, "complete", 0), cell(20, "complete", 1), cell(8, "ring", .5), cell(8, "ring", 1)]
    chosen += [cell(20, "sparse", rho, .3, .2) for rho in [0, .5, 1]]
    table_rows = []
    for row in chosen:
        R = "{} [{}, {}]".format(*(fmt(row[key]) for key in ["tail_R_mean_mean", "tail_R_mean_ci_low", "tail_R_mean_ci_high"]))
        table_rows.append([str(int(row["n_agents"])), row["topology"], fmt(row["rho"]), fmt(row["dropout"]), fmt(row["delay"]), R, fmt(row["tail_pairwise_disagreement_mean"])])
    output = r"\subsection{Separately executed synthetic oscillator extension}" + "\n"
    output += (
        f"The separate synthetic study executes {len(rows)} conditions with {sum(int(row['n']) for row in rows):,} independent network realizations: "
        r"$N\in\{3,8,20\}$, four topologies, correlation $\rho\in\{0,.5,1\}$, dropout $p\in\{0,.3\}$ and delay $\tau\in\{0,.2\}$. "
        r"Each uses $T=50$, $h=.01$, coupling 2 and marginal noise amplitude .35. Frequencies are a fixed shuffled grid from $-.5$ to $.5$ and initial phases are independently uniform. "
        "Dropout and delay use the separately recorded synthetic protocol; delayed coupling uses stale relative phase error.\n\n"
    )
    output += make_table(
        ["$N$", "Topology", r"$\rho$", "$p$", r"$\tau$", "Tail $R$ [CI]", "Disagreement"],
        table_rows, "rlrrrll", "Selected synthetic network results, $n=500$ networks per row. Tail summaries use $t\\ge40$. Intervals for tail $R$ are Student-based across independent networks. Disagreement is the tail mean of pairwise $1-\\cos(\\theta_i-\\theta_j)$. The three-agent ring and complete graph have identical adjacency, so they are not distinct topology evidence.", "tab:synthetic")
    clean, shared = cell(20, "complete", 0), cell(20, "complete", 1)
    ring_independent, ring_shared = cell(8, "ring", 0), cell(8, "ring", 1)
    output += (
        f"For the 20-agent complete graph without dropout/delay, tail concentration is {fmt(clean['tail_R_mean_mean'])} for independent noise and {fmt(shared['tail_R_mean_mean'])} for shared noise. "
        "Shared additive noise cancels exactly in phase differences: increasing correlation reduces differential noise at fixed marginal amplitude, rather than intrinsically synchronizing uncoupled agents. "
        "Topology comparisons depend on the degree-normalization convention and the specified frequency grid. "
        "These synthetic measurements assess oscillator coordination, not language-model correctness or communication-token savings.\n"
    )
    output += (
        f"The eight-agent clean ring is a counterexample to a monotone correlation--concentration interpretation: tail $R$ is {fmt(ring_independent['tail_R_mean_mean'])} at $\\rho=0$ and {fmt(ring_shared['tail_R_mean_mean'])} at $\\rho=1$, "
        f"while tail frequency dispersion falls from {fmt(ring_independent['tail_frequency_dispersion_mean'])} to {fmt(ring_shared['tail_frequency_dispersion_mean'])}. "
        "Low global concentration can therefore coexist with near-equal frequencies and a nonuniform locked phase profile. Diagnostic replays of existing paths investigate winding states; they are not counted as extra independent trajectories.\n"
    )
    if (ROOT / "report/figures/synthetic.pdf").exists():
        output += figure("synthetic", "Executed synthetic oscillator topology and communication comparisons. The phase-concentration observable belongs to the separately specified oscillator model and does not validate an LLM phase representation.", "fig:synthetic")
    return output, len(rows), sum(int(row["n"]) for row in rows)


def build() -> Path:
    families = load_conditions()
    grid, acquisition, long, sensitivity = (families[key] for key in ["grid", "acquisition", "long", "sensitivity"])
    convergence = convergence_rows()
    scenarios = {name: [row for row in convergence if row["scenario"] == name] for name in ["baseline", "near_threshold"]}
    metadata = read_json(ROOT / "results/run_metadata/full_run.json")
    environment = read_json(ROOT / "results/run_metadata/environment.json")
    replay = read_json(ROOT / "experiments/baseline/original_replay.json")
    baseline = find_cell(grid, .5, .15)
    near = find_cell(grid, .95, .15)
    closest = find_cell(grid, .99, .15)
    noisy = find_cell(grid, .9, .4)
    long_near = find_cell(long, .95, .15)
    long_zero = find_cell(long, .75, .15)
    long_high = find_cell(long, .9, .3)
    theory_variance = baseline["alpha"] ** 2 / (2 * baseline["a"] * baseline["b"] * math.sqrt(1 - baseline["delta"] ** 2))
    n_convergence = sum(int(rows[0]["n"]) for rows in scenarios.values())
    n_pll = sum(int(row["n"]) for rows in families.values() for row in rows) + n_convergence
    counts = [("E1--E4 grid (shared slices)", len(grid), sum(row["n"] for row in grid), "50"),
              ("Original-initial-state acquisition", 1, acquisition[0]["n"], "50"),
              ("E6 long-horizon ensembles", len(long), sum(row["n"] for row in long), "200"),
              ("E7 inertia/gain sensitivity", len(sensitivity), sum(row["n"] for row in sensitivity), "50"),
              ("E5 coupled mesh ensembles", len(scenarios), n_convergence, "10")]
    counts_table = make_table(
        ["Family", "Conditions", "Independent paths", "$T$"], counts,
        "lrrr", "Executed PLL experiment families. The grid includes nine deterministic controls; all other paths are stochastic. A coupled Brownian path is counted once, even though it is solved on five meshes and a reference mesh. The six original-seed replays are validation, not additional independent Monte Carlo evidence.", "tab:counts")
    chosen = [(.5, .15), (.5, .4), (.75, .3), (.9, .15), (.9, .3), (.9, .4), (.95, .15), (.99, .15)]
    result_rows = []
    for delta, alpha in chosen:
        row = find_cell(grid, delta, alpha)
        result_rows.append([fmt(delta), fmt(alpha), str(row["n"]), ci(row, "phi_variance"), ci(row, "locking_probability"), str(int(row["slips_total"]))])
    grid_table = make_table(
        [r"$\delta$", r"$\alpha$", "$n$", r"$\widehat{\mathrm{Var}}(\phi_T)$ [CI]", r"Capture [CI]", "Slips"],
        result_rows, "rrrlll", "Selected actual grid results at $T=50$, $h=0.001$. Variance has units rad$^2$; its 95\\% interval uses trajectory bootstrap resampling. Capture is a finite-horizon probability with a Wilson 95\\% interval. Slips are total committed transitions, not the number of paths with an event.", "tab:grid")
    long_rows = []
    for row in long:
        long_rows.append([fmt(row["delta"]), fmt(row["alpha"]), str(int(row["first_slip_events"])), str(int(row["slips_total"])), ci(row, "slip_rate"), fmt(row["rmst_first_slip"]), fmt(row["median_first_slip"])])
    long_table = make_table(
        [r"$\delta$", r"$\alpha$", "Event paths", "Slips", "Rate [CI]", "RMST", "Median"], long_rows,
        "rrrrlll", "Long-horizon first-slip and transition statistics: $n=1000$ per row and $T=200$. Rates are committed transitions per trajectory per model time; intervals bootstrap independent path rows. RMST is the restricted mean first-slip time through 200, including right censoring. An unavailable median means survival did not cross one half. Zero-event bootstrap intervals degenerate and require the separate upper bounds discussed in the text.", "tab:long")
    conv_rows = []
    for index in range(5):
        b, stress = scenarios["baseline"][index], scenarios["near_threshold"][index]
        def scaled_interval(row, keys):
            values = [f"{1000 * row[key]:.3g}" for key in keys]
            return f"{values[0]} [{values[1]}, {values[2]}]"
        conv_rows.append([
            f"{b['dt']:.6f}".rstrip("0"),
            scaled_interval(b, ["strong_rms", "strong_low", "strong_high"]),
            scaled_interval(stress, ["strong_rms", "strong_low", "strong_high"]),
            scaled_interval(stress, ["mean_bias", "mean_bias_low", "mean_bias_high"]),
        ])
    convergence_table = make_table(
        ["Step $h$", r"\shortstack{Baseline RMS\\$\times10^{-3}$ [CI]}", r"\shortstack{Stress RMS\\$\times10^{-3}$ [CI]}", r"\shortstack{Stress phase bias\\$\times10^{-3}$ [CI]}"], conv_rows,
        "rlll", "Coupled endpoint state errors relative to the numerical reference $h_{\\rm ref}=0.00015625$, $T=10$. All three estimate/interval columns are in units of $10^{-3}$. Baseline: $n=1000$, $(\\delta,\\alpha)=(0.5,0.15)$; stress: $n=500$, $(0.95,0.4)$. Each state error combines phase and velocity in the declared model units. Intervals quantify Monte Carlo uncertainty, not reference discretization error. Phase bias is a paired weak difference.", "tab:convergence")
    unlocked_rows = []
    for delta, alpha in [(1.01, .15), (1.05, .15), (1.05, .4)]:
        row = find_cell(grid, delta, alpha)
        unlocked_rows.append([fmt(delta), fmt(alpha), str(row["n"]), ci(row, "tail_velocity_mean"), fmt(row["rotation_rate"])])
    unlocked_table = make_table(
        [r"$\delta$", r"$\alpha$", "$n$", "Tail velocity [CI]", "Rotation rate"], unlocked_rows,
        "rrrll", "Above-threshold dynamics at $T=50$. Tail velocity is in rad/model-time; rotation rate is tail velocity divided by $2\\pi$, in cycles/model-time. Equilibrium-based capture and slip columns are not applicable.", "tab:unlocked")
    sensitivity_rows = []
    for row in sensitivity:
        if math.isclose(row["alpha"], .15):
            sensitivity_rows.append([fmt(row["a"]), fmt(row["b"]), fmt(row["delta"]), fmt(row["phi_variance"]), fmt(row["tail_velocity_mean"]), fmt(row["locking_probability"])])
    sensitivity_table = make_table(
        ["$a$", "$b$", r"$\delta$", r"$\widehat{\mathrm{Var}}(\phi_T)$", "Tail velocity", "Capture"], sensitivity_rows,
        "rrrrrr", "E7 at fixed $\\Delta=0.5$, $\\alpha=0.15$, $n=500$, $T=50$, $h=0.001$. The second noise level $\\alpha=0.3$ is also executed and retained in the machine-readable table. Changing $a,b$ changes normalized damping, detuning and noise simultaneously. Dashes denote absence of an ordinary stable well.", "tab:sensitivity")

    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\usepackage{microtype}
\makeatletter
\renewenvironment{thebibliography}[1]
{\section{Bibliography}\list{\@biblabel{\@arabic\c@enumiv}}
{\usecounter{enumiv}\setlength{\leftmargin}{2em}\setlength{\labelwidth}{1.5em}\setlength{\labelsep}{.5em}}\sloppy}
{\endlist}
\makeatother
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{1em}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\renewcommand{\topfraction}{.92}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{.07}
\renewcommand{\floatpagefraction}{.8}
\title{Stochastic Phase-Locked Loops:\\Numerical Validation, Lock Retention and Noise-Induced Slips}
\author{}
\date{13 November 2026}
\begin{document}
\maketitle
\begin{abstract}
"""
    sections = [preamble]
    sections.append(
        "We investigate a textbook second-order phase-locked loop under additive Gaussian white noise. "
        f"The executed study comprises {n_pll - 9:,} independent PLL Brownian trajectories and nine deterministic controls across a full noise--detuning grid, a separate acquisition ensemble, longer first-passage experiments, inertia/gain sensitivity and two coupled numerical-resolution ensembles. "
        f"At the baseline $(\\delta,\\alpha)=(0.5,0.15)$, terminal phase variance is {fmt(baseline['phi_variance'])} rad$^2$, with no committed slips observed among {baseline['n']} paths by $T=50$. "
        f"At $(0.95,0.15)$, {int(near['slips_total'])} transitions occur over the same horizon and {int(long_near['slips_total'])} occur in a separate $T=200$ ensemble. "
        "Coupled meshes support approximately first-order endpoint strong error relative to a finer numerical solution. "
        "All locking and slip statements are finite-horizon operational outcomes; neither permanent locking nor a physical hardware reproduction is claimed.\n"
        r"\end{abstract}" + "\n"
    )
    sections.append(r"""
\section{Motivation and Background}
Phase-locked loops adjust an oscillator to track a reference phase. Noise perturbs this feedback process and can produce small fluctuations, loss of finite-time capture, or completed changes by $2\pi$ in unwrapped phase. A deterministic attracting equilibrium does not specify how often noisy paths leave its neighborhood. This project asks how detuning and noise alter these outcomes, how mesh size affects the numerical answer, and what information is lost by counting temporary threshold crossings as slips.

The starting point is the communications example in Kloeden and Platen, Chapter~7, p.~274, equation~(10.4) \cite{kloeden1992}. The supplied page image was inspected directly. The original Python script already implements the correct Euler--Maruyama recurrence under its stated unit-Wiener convention. Its five illustrative stochastic paths do not constitute an uncertainty or rare-event study. We preserve that source and reproduce its six seed settings separately, then add independent ensembles, explicit event definitions, trajectory-level uncertainty and convergence analysis. No measured hardware dataset or physical time calibration is available; the data below are simulations of the stated model.

\section{Brief Literature Survey}
Kloeden and Platen provide the model source and a general numerical SDE framework \cite{kloeden1992}. Higham distinguishes Euler--Maruyama, Milstein, strong and weak errors, and numerical stability \cite{higham2001}. These distinctions motivate coupled Brownian paths for mesh comparisons and prevent treating Monte Carlo spread as discretization error.

Bobrovsky and Schuss describe second-order PLL cycle slipping as a Kolmogorov exit problem \cite{bobrovsky1982}; their publisher abstract and metadata were verified, but their full asymptotic formula is not used here. Kramers' primary barrier-escape analysis provides the historical setting for noise-assisted transitions and damping-dependent rates \cite{kramers1940}. Khasminskii distinguishes moment, almost-sure and distributional questions of stochastic stability \cite{khasminskii2012}. No theorem number from an inaccessible text is claimed as a validation result.

Bianchi et al. demonstrate solver- and initialization-dependent errors in a different, lead--lag-filter PLL \cite{bianchi2015}. Their examples support caution about inferring global capture from a few trajectories, without establishing hidden attractors in this project. The Kuramoto review motivates a separately defined synthetic network of phase oscillators \cite{acebron2005}. Sparse LLM debate is an empirical communication baseline \cite{li2024}; it supplies no derivation that language-model messages obey PLL dynamics. GSM8K is a possible later benchmark \cite{cobbe2021,gsm8krepository}. This targeted source-checked survey is not an exhaustive statement of all literature through 2026.

\section{Model Description}
\subsection{Source equation and stochastic convention}
Write detuning as $\Delta=\omega-\omega_0$. The verified equation is
\begin{equation}
a\ddot\phi+\dot\phi+ab\sin\phi=\Delta-\alpha\zeta_t,
\qquad a>0,\ b>0,\ \alpha\ge0.
\label{eq:second}
\end{equation}
The source identifies $a=RC$ as the linear-filter time constant and relates $b$ to the signal-amplitude product. We explicitly assume $\zeta_t\,dt=dW_t$, where $W$ is standard Brownian motion, so $\mathbb E[(W_{t+h}-W_t)^2]=h$. The source calls $\alpha$ noise ``intensity'' without separately fixing that covariance normalization. In this project $\alpha$ is the amplitude multiplying unit white noise; variance rate is proportional to $\alpha^2$.

With phase velocity $v=\dot\phi$, the It\^o state system is
\begin{equation}
d\phi=v\,dt,\qquad
dv=\frac{\Delta-v-ab\sin\phi}{a}\,dt-\frac{\alpha}{a}\,dW_t.
\label{eq:sde}
\end{equation}
Its drift is $F=(v,(\Delta-v-ab\sin\phi)/a)^\mathsf T$ and diffusion is the constant vector $G=(0,-\alpha/a)^\mathsf T$. Both are globally Lipschitz for fixed valid parameters. Noise acts directly on velocity and reaches phase through integration. It\^o and Stratonovich interpretations coincide because the diffusion derivative vanishes. The negative sign is retained for reproducible paths; reversing the Brownian sign leaves the marginal law unchanged.

Time is in model units, phase in radians and velocity in radians/model-time. Under physical seconds, dimensional consistency would require $[a]={\rm s}$, $[b]={\rm s}^{-2}$, $[\Delta]={\rm s}^{-1}$ and $[\alpha]={\rm s}^{-1/2}$. For $\tau=\sqrt b\,t$ and $u=v/\sqrt b$, the normalized equation is
\begin{equation}
d\phi=u\,d\tau,\quad du=(\delta-\gamma u-\sin\phi)d\tau-\beta\,dB_\tau,
\quad
\delta=\frac{\Delta}{ab},\quad \gamma=\frac1{a\sqrt b},\quad
\beta=\frac{\alpha}{ab^{3/4}}.
\end{equation}
Thus normalized detuning alone does not determine damping or noise.

\subsection{Equilibria and local stability}
For $\alpha=0$ and $|\delta|<1$, let $\theta=\arcsin\delta$. The stable branch and saddle branch are
\begin{equation}
(\phi_{s,k},v_s)=(\theta+2k\pi,0),\qquad
(\phi_{u,k},v_u)=(\pi-\theta+2k\pi,0).
\end{equation}
The Jacobian and characteristic equation are
\begin{equation}
A_* =\begin{pmatrix}0&1\\-b\cos\phi_*&-1/a\end{pmatrix},\qquad
\lambda^2+\lambda/a+b\cos\phi_*=0.
\end{equation}
On the stable branch the trace is negative and determinant positive, giving two eigenvalues with negative real parts. The saddle has negative determinant. At $|\delta|=1$ one eigenvalue is zero; at $|\delta|>1$ no equilibrium exists. The strict inequality is therefore an equilibrium-existence and local-attraction condition, not a theorem of global capture from every initial phase and velocity. For nonzero noise the deterministic equilibrium is not absorbing.

\subsection{Potential, barriers and fluctuations}
Let $K=ab$ and define
\begin{equation}
U(\phi)=-\Delta\phi-K\cos\phi,\qquad H=av^2/2+U(\phi).
\end{equation}
Deterministically, $dH/dt=-v^2$. It\^o's formula instead gives
\begin{equation}
dH=\left(-v^2+\frac{\alpha^2}{2a}\right)dt-\alpha v\,dW_t.
\end{equation}
Noise injects energy in expectation. Since $U(\phi+2\pi)=U(\phi)-2\pi\Delta$, the potential is tilted and is not globally confining on unwrapped phase. The downhill barrier is
\begin{equation}
\Delta U_{\rm down}=2K\left[\sqrt{1-\delta^2}-|\delta|\arccos|\delta|\right],
\end{equation}
which vanishes as $(1-|\delta|)^{3/2}$ near the threshold. This motivates the near-threshold conditions; it is not an exact escape-rate formula. The local linearized Gaussian model predicts
\begin{equation}
\operatorname{Var}(v)=\frac{\alpha^2}{2a},\qquad
\operatorname{Var}(\phi-\phi_s)=\frac{\alpha^2}{2ab\cos\phi_s}.
\label{eq:localvar}
\end{equation}
These are local, no-slip approximations and become unreliable near vanishing barriers.

\subsection{Discretization and completed transitions}
For $h=\Delta t$ and independent $Z_n\sim N(0,1)$,
\begin{align}
\Delta W_n&=\sqrt h\,Z_n,\\
\phi_{n+1}&=\phi_n+v_nh,\\
v_{n+1}&=v_n+\frac{\Delta-v_n-ab\sin\phi_n}{a}h-\frac{\alpha}{a}\Delta W_n.
\end{align}
Both updates use the old state. The standard vector Milstein correction is $\tfrac12(DG\,G)[(\Delta W_n)^2-h]=0$, so Milstein and Euler--Maruyama give the same recurrence here. Labeling the method Milstein cannot improve its accuracy.

A primary committed slip requires entry within $\pi/4$ of an adjacent stable center and continuous residence there for 0.5 model time units. The occupied well index then changes; temporary exits and recrossings are not automatically events. Loose $(\pi/2,0.25)$ and strict $(\pi/6,1)$ variants test detector sensitivity. Stable-well cores are operational regions, not exact two-dimensional basin boundaries. A local saddle stable-manifold approximation is used only for geometry. At $|\delta|\ge1$ stable-well capture and slips are not applicable; mean rotation is reported instead.

\section{Recent Developments / State of the Art}
The relevant methodological direction is to combine nonlinear dynamics, reproducible stochastic numerics and explicit uncertainty instead of relying on individual simulation traces. The present implementation contributes an auditable experiment package for this teaching model; it does not introduce a new SDE integrator or prove a new PLL theorem. Large-horizon first-passage sampling and sensitivity to operational event definitions expose limitations of short illustrative runs.

For the synthetic extension, define network phases explicitly, for example
\begin{equation}
d\theta_i=\left[\omega_i+K_c\sum_jw_{ij}\sin(\theta_j-\theta_i)\right]dt
+\sigma\,dB_i,
\qquad R=\left|\frac1N\sum_i e^{i\theta_i}\right|.
\end{equation}
The order parameter $R$ measures phase concentration; finite small networks can show nonzero $R$ by chance. Topology, correlated noise and communication delay/dropout belong to a separate model definition. Synchronization does not establish correctness of a reasoning system.

An LLM extension remains conditional on defining a phase representation and showing incremental predictive value over semantic/answer disagreement on development data. An embedding is not intrinsically a physical angle. Fixed sparse communication from Li et al.\ \cite{li2024} is a necessary baseline, alongside independent agents, full communication and disagreement-triggered communication. Full official GSM8K test evaluation, separate tuning data, prompt/model controls, call-matched comparisons and recorded cost are required before interpreting gains. No paid API inference is part of the local PLL evidence.

A September 2026 preprint by Wang et al.\ \cite{wang2026} reports strong distinct-peer random routing and lightweight stopping baselines. Its date, authors and abstract were verified; it is preprint-level evidence, not peer-reviewed confirmation. It further motivates testing simple routing/stopping before attributing gains to a more complex controller.

\section{Results and Discussion}
\subsection{Executed design and statistical units}
""")
    sections.append(counts_table)
    sections.append(
        r"The grid crosses the noise amplitudes and normalized detunings"
        r"\[\begin{aligned}\alpha&\in\{0,.025,.05,.10,.15,.20,.30,.40\},\\\delta&\in\{0,.25,.50,.75,.90,.95,.99,1.01,1.05\},\end{aligned}\]"
        r"at $a=b=1$, $T=50$, $h=.001$. "
        "The nine zero-noise cells use one deterministic control; all 63 stochastic cells use at least 500 independent trajectories, with 1000 in selected baseline/near-threshold comparisons. "
        r"Grid paths start at $(\arcsin\delta,0)$ where a stable branch exists, and at $(0,0)$ above threshold. The separate acquisition ensemble starts at $(\arcsin(.5)+.5,0)$. "
        "E1--E4 are views of the same grid, not extra replications.\n\n"
        "Each stochastic summary uses independent trajectories as its statistical units. Pointwise mean intervals are Student-based; variance and slip-rate intervals resample path rows 2000 times. "
        r"Capture uses $|\operatorname{wrap}(\phi_T-\phi_s)|<\pi/4$ and absolute mean velocity over the last 20\% of the horizon below 0.1. Retained lock also requires no committed slip. Binary intervals use Wilson's method. "
        "No significance claims are based on unadjusted multiple comparisons. A confidence band for the mean is not a 95\\% path-containment band.\n\n"
    )
    sections.append(r"\subsection{Baseline, noise and detuning}" + "\n")
    sections.append(grid_table)
    sections.append(
        f"At the baseline, terminal phase variance is {ci(baseline, 'phi_variance')} rad$^2$, compared with the local linear prediction {fmt(theory_variance)} rad$^2$. "
        f"All {baseline['n']} paths satisfy terminal capture, with Wilson interval [{fmt(baseline['locking_probability_low'])}, {fmt(baseline['locking_probability_high'])}]. "
        f"No committed transition is observed; the Wilson upper bound on the probability of at least one event by 50 is {fmt(baseline['slip_probability_high'])}. "
        "This is evidence of finite-time retention under this initialization, not proof of zero slip probability.\n\n"
        f"At the same noise amplitude and $\\delta=.95$, variance rises to {fmt(near['phi_variance'])} rad$^2$ and {int(near['slips_total'])} committed transitions occur. "
        f"At $\\delta=.99$ it is {fmt(closest['phi_variance'])} rad$^2$, with capture {ci(closest, 'locking_probability')}. "
        f"For $(\\delta,\\alpha)=(.9,.4)$, {int(noisy['slips_total'])} transitions occur and capture is {ci(noisy, 'locking_probability')}. "
        "These nonlinear mixtures of well occupancy and displacement cannot be described by the local Gaussian variance alone.\n"
    )
    sections.append(figure("baseline", "Executed baseline deterministic/stochastic trajectories, ensemble moments and phase-space view. The stable-equilibrium initialization and original acquisition initialization are kept distinct; displayed uncertainty is labeled by its statistical meaning.", "fig:baseline"))
    sections.append(figure("combined_grid", "Full executed noise--detuning matrix. Near-threshold fluctuations and transitions depend jointly on detuning and noise. Cells without stable equilibria must be interpreted through rotation rather than ordinary locked-state slips.", "fig:grid"))
    sections.append(unlocked_table)
    sections.append(
        "Above threshold, positive mean rotation remains a useful observable while stable-well slip labels cease to apply (Table~\\ref{tab:unlocked}). "
        f"The separate acquisition ensemble has phase variance {ci(acquisition[0], 'phi_variance')} rad$^2$ at $T=50$ and {int(acquisition[0]['slips_total'])} observed committed slips. "
        f"Replays of seeds 0--4 and 42 agree with the preserved implementation to maximum phase difference {fmt(max(row['max_phase_difference'] for row in replay))} rad and velocity difference {fmt(max(row['max_velocity_difference'] for row in replay))} rad/model-time. "
        "Floating-point roundoff agreement validates recurrence preservation, not the physical noise model.\n"
    )
    sections.append(r"\subsection{Long horizons, first passage and detector dependence}" + "\n")
    sections.append(long_table)
    sections.append(
        f"At $(\\delta,\\alpha)=(.95,.15)$, the independent $T=200$ ensemble records {int(long_near['first_slip_events'])}/{long_near['n']} paths with a first event and {int(long_near['slips_total'])} total transitions. "
        f"Its restricted mean first-slip time is {fmt(long_near['rmst_first_slip'])}, whereas the mean restricted to observed-event paths is {fmt(long_near['conditional_first_slip_mean'])}. "
        "The latter omits censored paths and is not the unconditional mean waiting time. Kaplan--Meier survival retains right censoring; no extrapolation beyond 200 is made.\n\n"
        f"The $(.75,.15)$ long ensemble has no observed events. Its Wilson event-probability upper bound is {fmt(long_zero['slip_probability_high'])}; "
        f"a separately labeled Poisson exposure approximation gives a rate upper bound {fmt(long_zero['zero_event_rate_upper_poisson'])} per model time. "
        "A degenerate zero-event bootstrap interval is not evidence of an exactly zero rate.\n\n"
        f"At $(.9,.3)$ the primary rate is {fmt(long_high['slip_rate'])}, the loose rate {fmt(long_high['loose_slip_rate'])}, and the strict rate {fmt(long_high['strict_slip_rate'])}. "
        "This sensitivity shows that completed recapture and rapid running excursions are distinguished by the detector. The primary detector does not claim to count every engineering notion of a phase slip.\n"
    )
    sections.append(figure("slip_survival", "Actual long-horizon first-slip survival estimates. Unslipped paths are censored at the horizon; event-time uncertainty and operational detector dependence limit rare-event interpretation.", "fig:survival"))
    sections.append(r"\subsection{Numerical convergence and parameter sensitivity}" + "\n")
    sections.append(convergence_table)
    sections.append(
        f"All saved convergence states are finite. The fitted log--log endpoint RMS slopes are {fmt(scenarios['baseline'][0]['rate_fit'])} for the baseline and {fmt(scenarios['near_threshold'][0]['rate_fit'])} for the stress case. "
        "These are empirical rates over the tested meshes relative to a finer numerical reference, not exact-solution error certificates. "
        "Coarse increments sum the same underlying fine increments, so differences compare discretizations of the same forcing. "
        "Weak phase, phase-square and variance differences are stored separately, and no rate is forced on unresolved noisy weak errors. "
        "The finite $T=10$ endpoint study does not establish convergence of $T=200$ rare-event rates or of basin boundaries.\n"
    )
    sections.append(figure("convergence", "Coupled Brownian mesh comparisons for the baseline and nonlinear near-threshold stress scenario. The reference itself has finite step size; slopes describe the tested error window.", "fig:convergence"))
    sections.append(sensitivity_table)
    sections.append(
        "The E7 results separate no-equilibrium/critical cases from ordinary stable wells. Larger $ab$ at fixed detuning moves the system farther from the equilibrium threshold, while $a$ also changes damping and diffusion in the original units. "
        "Consequently the table cannot identify a single standardized causal effect of inertia independently of the other dimensionless groups. "
        "The full second noise level and trajectory-level summaries remain available in the CSV/JSON archive.\n"
    )
    synthetic_text, synthetic_conditions, synthetic_n = synthetic_results()
    sections.append(synthetic_text)
    sections.append(r"\subsection{Reproducibility and limitations}" + "\n")
    test_log = (ROOT / "results/run_metadata/preflight_tests.txt").read_text().strip().splitlines()[-1]
    final_test_path = ROOT / "results/run_metadata/final_tests.txt"
    final_tests = ""
    if final_test_path.exists():
        final_tests = "The full post-run test record reports \\texttt{" + escape(final_test_path.read_text().strip().splitlines()[-1]) + "}. "
    packages = environment["packages"]
    sections.append(
        "The preflight record reports \\texttt{" + escape(test_log) + "}. "
        + final_tests
        +
        "Checks cover reproducibility, Brownian scaling, deterministic agreement, equilibrium classification, finite states, completed transitions and statistical calculations. "
        "The execution manifest, source fingerprints, per-condition configuration, seeds, trajectory rows and sampled paths are preserved. "
        f"The recorded environment uses Python {escape(environment['python'].split()[0])}, NumPy {escape(packages['numpy'])}, SciPy {escape(packages['scipy'])}, pandas {escape(packages['pandas'])} and Matplotlib {escape(packages['matplotlib'])}. "
        "Master seed 20261113 and condition/trajectory identifiers initialize independent reproducible streams.\n\n"
        r"From the project root, execute \texttt{python -m src.reproduce}; \texttt{--reports-only} regenerates reports from saved results. The individual validation command is \texttt{python -m pytest -q}; granular workflows are documented in \texttt{README.md}. "
        r"The formal report is rebuilt by \texttt{python -m src.latex\_report}; compile inside \texttt{report/} using \texttt{tectonic main.tex}, or two \texttt{pdflatex main} passes with the required packages installed. Its explicit bibliography is generated from the eleven verified reusable \texttt{references.bib} records, so a separate BibTeX executable is unnecessary. "
        "The HTML report contains the extended derivations and code walkthrough; this concise report retains the scientific model, measured results and their limits.\n\n"
        "The principal limits are synthetic rather than measured input, uncalibrated time/noise, finite horizon and sample size, operational detector thresholds, a finite numerical reference, and no proof of global capture or rare-event asymptotics. "
        "Pointwise and per-cell intervals do not supply familywise simultaneous confidence. Comparisons are descriptive and mechanism-informed; no unsupported novelty or significance claim is made.\n"
    )
    sections.append(r"""
\section{Conclusion and Possible Extensions}
The textbook equation supports a reproducible, statistically explicit study that goes beyond a few noisy traces. The saved ensembles distinguish small local fluctuations from large unwrapped displacements, show why near-threshold and longer-horizon studies matter, and separate above-threshold rotation from equilibrium-based slips. Standard Milstein and Euler--Maruyama coincide for this additive-noise model; coupled results support approximately first-order endpoint behavior over the tested meshes without proving rare-event convergence.

Further work should calibrate a physical loop and noise spectrum, validate event definitions against application-specific requirements, compare higher-order/integrated-noise methods, and resolve very rare transitions with justified longer horizons or validated rare-event algorithms. A synthetic oscillator network is a separate mathematical extension. LLM communication requires an independently validated coordination signal, full benchmark controls and approved costs before any claim about reasoning performance. No physical phase interpretation or benchmark improvement follows from synchronization alone.

\clearpage
""")
    sections.append(embedded_bibliography())
    sections.append(r"\end{document}" + "\n")
    content = "".join(sections)
    if synthetic_n:
        content = content.replace(
            r"\end{abstract}",
            f"A separate {synthetic_n:,}-network study compares communication topologies, correlation, dropout and delay, and distinguishes phase concentration from frequency synchronization.\n" + r"\end{abstract}",
        )
    dataset_path = ROOT / "experiments/llm_extension/dataset_manifest.json"
    if dataset_path.exists():
        dataset = read_json(dataset_path)
        n_train, n_test = dataset["train"]["n"], dataset["test"]["n"]
        logical_calls = 5 * 7 * n_test
        llm_plan = (
            f"The downloaded official GSM8K files contain {n_train:,} training and {n_test:,} test questions, with SHA-256 hashes retained in the dataset manifest. "
            f"A proposed five-method, seven-call-per-method base evaluation would require {logical_calls:,} logical calls for one full test pass. "
            f"At planning assumptions of 2000 input and 800 output tokens per call, that is {logical_calls * 2000 / 1e6:.3f} million input and {logical_calls * 800 / 1e6:.3f} million output tokens; these are estimates, not observed usage. "
            "No deployment-specific dollar cost is asserted. The phase-signal gate remains unmet, so the full paid experiment is deferred and no API calls are executed.\n\n"
        )
        marker = r"\section{Results and Discussion}"
        if content.count(marker) != 1:
            raise ValueError("Expected one results-section boundary")
        content = content.replace(marker, llm_plan + marker)
    output = ROOT / "report/main.tex"
    output.write_text(content)
    print(f"Wrote {output}; {n_pll:,} PLL trajectories/control paths; {synthetic_conditions} synthetic conditions/{synthetic_n:,} networks; source status={metadata.get('status', 'unknown')}")
    return output


if __name__ == "__main__":
    build()
