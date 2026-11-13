
import numpy as np
import matplotlib.pyplot as plt


def simulate_pll(
    a=1.0,
    b=1.0,
    detuning=0.5,
    noise=0.15,
    T=50.0,
    dt=0.001,
    phi0=None,
    v0=0.0,
    seed=42,
):
    """
    Simulate the textbook PLL equation using Euler-Maruyama.

    a * phi'' + phi' + a*b*sin(phi)
        = detuning - alpha * white_noise

    Assumption: white noise is normalized so that
    zeta_t dt = dW_t, with standard Brownian motion W.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive.")
    if noise < 0 or dt <= 0 or T <= 0:
        raise ValueError("noise must be nonnegative; dt and T positive.")

    n_steps = int(round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # A locked equilibrium exists for |detuning| < a*b.
    ratio = detuning / (a * b)
    if abs(ratio) < 1:
        phi_star = np.arcsin(ratio)
    else:
        phi_star = 0.0  # Illustrative initialisation if unlocked.

    if phi0 is None:
        phi0 = phi_star + 0.5

    phi = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    phi[0], v[0] = phi0, v0

    rng = np.random.default_rng(seed)
    dW = np.sqrt(dt) * rng.standard_normal(n_steps)

    for n in range(n_steps):
        # Phase evolves according to phase velocity.
        phi[n + 1] = phi[n] + v[n] * dt

        # Phase velocity follows the stochastic PLL equation.
        drift = (
            detuning - v[n] - a * b * np.sin(phi[n])
        ) / a

        v[n + 1] = (
            v[n]
            + drift * dt
            - (noise / a) * dW[n]
        )

    return t, phi, v, phi_star


def plot_pll_dynamics(t, phi_det, v_det, paths, phi_star, params, save_path="stochastic_pll_dynamics.png"):
    """
    Generate an enhanced, publication-grade visualization of PLL dynamics.
    Includes time-series of phase and velocity, dual degree/radian axes,
    ensemble mean and variance band, and the 2D phase portrait (v vs phi).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 8), layout="constrained", dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, 1])

    # Curated color palette
    det_color = "#0F172A"
    palette = ["#0284C7", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]

    # 1. Phase Dynamics Subplot
    ax0.axvspan(0, 8, color="#F1F5F9", alpha=0.9, label="Transient Acquisition")
    ax0.axhline(
        phi_star,
        color="#DC2626",
        linestyle="--",
        linewidth=1.6,
        label=rf"Stable Lock $\phi^* = \arcsin(\Delta/ab) \approx {phi_star:.3f}$ rad ({np.degrees(phi_star):.1f}$^\circ$)",
    )

    phi_stack = np.array([p[1] for p in paths])
    phi_mean = np.mean(phi_stack, axis=0)
    phi_std = np.std(phi_stack, axis=0)
    ax0.fill_between(
        t,
        phi_mean - phi_std,
        phi_mean + phi_std,
        color="#0284C7",
        alpha=0.15,
        label=r"Ensemble Mean $\pm 1\sigma$",
    )

    for i, (_, phi, _, _) in enumerate(paths):
        ax0.plot(
            t,
            phi,
            color=palette[i % len(palette)],
            alpha=0.55,
            linewidth=1.1,
            label=f"Stochastic Run {i+1}",
        )

    ax0.plot(t, phi_det, color=det_color, linewidth=2.4, label="Deterministic (Noise = 0)")
    ax0.set_ylabel(r"Unwrapped Phase $\phi(t)$ [rad]", fontsize=11)
    ax0.set_title(
        "Second-Order Phase-Locked Loop (PLL) Dynamics under Gaussian Noise",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax0.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5, ncol=2)
    ax0.set_ylim(0.05, 1.25)

    # Secondary twin axis for phase in degrees
    ax0_deg = ax0.twinx()
    ax0_deg.set_ylim(np.degrees(0.05), np.degrees(1.25))
    ax0_deg.set_ylabel("Phase [deg]", fontsize=10, color="#64748B")
    ax0_deg.tick_params(colors="#64748B")
    ax0_deg.grid(False)

    # 2. Phase Velocity Subplot
    ax1.axvspan(0, 8, color="#F1F5F9", alpha=0.9)
    ax1.axhline(0.0, color="#DC2626", linestyle="--", linewidth=1.6, label=r"Locked Velocity ($v^* = 0$)")

    for i, (_, _, v, _) in enumerate(paths):
        ax1.plot(t, v, color=palette[i % len(palette)], alpha=0.55, linewidth=1.0)
    ax1.plot(t, v_det, color=det_color, linewidth=2.4, label="Deterministic Velocity")

    ax1.set_xlabel(r"Time $t$ [dimensionless units]", fontsize=11)
    ax1.set_ylabel(r"Phase Velocity $v(t) = \dot{\phi}(t)$ [rad/s]", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)
    ax1.set_xlim(0, t[-1])

    # 3. Phase Portrait Subplot (v vs phi)
    ax2.plot(phi_det, v_det, color=det_color, linewidth=2.5, label="Deterministic Trajectory", zorder=4)
    for i, (_, phi, v, _) in enumerate(paths):
        ax2.plot(phi, v, color=palette[i % len(palette)], alpha=0.35, linewidth=0.8, zorder=2)

    ax2.scatter(
        [phi_det[0]],
        [v_det[0]],
        color="#2563EB",
        s=70,
        zorder=5,
        label=rf"Start $(\phi_0, v_0) = ({phi_det[0]:.2f}, 0)$",
    )
    ax2.scatter(
        [phi_star],
        [0.0],
        color="#DC2626",
        s=110,
        marker="*",
        zorder=6,
        label=rf"Attractor $(\phi^*, 0) = ({phi_star:.2f}, 0)$",
    )

    param_text = (
        "Parameters:\n"
        rf"$a = {params['a']}$ (Filter inertia)" "\n"
        rf"$b = {params['b']}$ (Loop gain)" "\n"
        rf"$\Delta = {params['detuning']}$ (Detuning)" "\n"
        r"$\alpha = 0.15$ (Noise intensity)" "\n"
        rf"$dt = 10^{{-3}}, \, T = {int(params['T'])}$"
    )
    ax2.text(
        0.05,
        0.05,
        param_text,
        transform=ax2.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )

    ax2.set_title(r"Phase Portrait ($v$ vs $\phi$)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel(r"Phase $\phi$ [rad]", fontsize=11)
    ax2.set_ylabel(r"Phase Velocity $v$ [rad/s]", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Enhanced plot successfully saved to: {save_path}")

    return fig


def main():
    params = dict(
        a=1.0,
        b=1.0,
        detuning=0.5,
        T=50.0,
        dt=0.001,
    )

    # Deterministic baseline: no random perturbation.
    t, phi_det, v_det, phi_star = simulate_pll(
        **params, noise=0.0, seed=42
    )

    # Multiple stochastic trajectories.
    paths = [
        simulate_pll(**params, noise=0.15, seed=seed)
        for seed in range(5)
    ]

    output_path = "stochastic_pll_dynamics.png"
    plot_pll_dynamics(t, phi_det, v_det, paths, phi_star, params, save_path=output_path)
    plt.show()


if __name__ == "__main__":
    main()

