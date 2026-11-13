# Mathematical model and numerical interpretation

## 1. Verified source and project conventions

The supplied image `notes/textbook_p274.png` was read visually. Under “Communications”, Kloeden and Platen, *Numerical Solution of Stochastic Differential Equations*, Chapter 7, p.274, equation (10.4), gives

\[
a\ddot\phi+\dot\phi+ab\sin\phi=\omega-\omega_0-\alpha\zeta_t. \tag{1}
\]

The accompanying paragraph identifies `phi` as the phase difference between the input and reference signals, `a=RC` as the linear-filter time constant, `b` as related to the product of the two signal amplitudes, and `alpha` as the intensity of white noise. It describes cycle slips as phase changes by `2pi` and proposes studying the relation between mean phase-change rate and detuning. The verified book metadata are available from [Springer](https://link.springer.com/book/10.1007/978-3-662-12616-5).

We write `Delta=omega-omega_0`, assume `a>0`, `b>0`, `alpha>=0`, and explicitly normalize white noise by

\[
\zeta_t\,dt=dW_t,\qquad
\mathbb E[(W_{t+h}-W_t)^2]=h. \tag{2}
\]

This Wiener normalization is a project assumption consistent with the original code. The supplied PLL paragraph calls alpha an intensity without fixing a separate covariance convention. Here **alpha is the coefficient of unit white noise**, so its squared value, not alpha itself, controls covariance. If another convention uses `E[zeta_t zeta_s]=2D delta(t-s)`, its force coefficient is converted through the product `alpha sqrt(2D)`. Calling a spectral density, variance rate, and amplitude by the same word “intensity” would change the simulated process.

The sign in (1) is preserved. Replacing `W` by `-W` leaves its law unchanged, so the sign does not change a single-condition distribution. It changes a fixed-seed path and must remain consistent in exact replays and coupled comparisons.

## 2. Why the second-order expression is an SDE

A standard Brownian motion starts at zero, has independent Gaussian increments, and has continuous paths. Over a step of duration `h`,

\[
\Delta W_n=W_{t_{n+1}}-W_{t_n}=\sqrt h\,Z_n,
\qquad Z_n\overset{\mathrm{iid}}\sim N(0,1). \tag{3}
\]

The factor `sqrt(h)` ensures variance `h`; using `h Z_n` would give variance `h^2` and drive the total noise variance to zero as the mesh is refined. White noise is the generalized derivative of Brownian motion, not an ordinary function that can be evaluated pointwise. Thus `phi''` in (1) is a shorthand interpreted through an integral equation.

Set `v=dphi/dt`. Integrate the force balance, divide by `a`, and obtain

\[
\begin{aligned}
d\phi_t&=v_t\,dt,\\
dv_t&=\left(\frac{\Delta}{a}-\frac{v_t}{a}-b\sin\phi_t\right)dt
-\frac{\alpha}{a}\,dW_t. \tag{4}
\end{aligned}
\]

Equivalently,

\[
\phi_t=\phi_0+\int_0^t v_s\,ds,\quad
v_t=v_0+\int_0^t\frac{\Delta-v_s-ab\sin\phi_s}{a}\,ds
-\frac{\alpha}{a}W_t. \tag{5}
\]

Velocity is continuous but, for nonzero noise, generally not differentiable. Phase is continuously differentiable with derivative velocity; the formal acceleration contains white noise. Noise acts directly on velocity and influences phase through integration.

For state `X=(phi,v)^T`, drift and diffusion are

\[
F(X)=\begin{pmatrix}v\\(\Delta-v-ab\sin\phi)/a\end{pmatrix},\qquad
G=\begin{pmatrix}0\\-\alpha/a\end{pmatrix},\quad
GG^T=\begin{pmatrix}0&0\\0&\alpha^2/a^2\end{pmatrix}. \tag{6}
\]

The sine function has bounded derivative and the other terms are linear. Consequently the drift is globally Lipschitz with linear growth for fixed valid parameters; the constant diffusion is also globally Lipschitz. These properties provide existence and uniqueness of a global strong solution for finite initial data. Diffusion is rank one in this two-dimensional state; independent noise must not be added to the phase equation without changing the model.

Because `G` is constant, the Itô-to-Stratonovich drift correction is zero. Both interpretations define the same process here. For a smooth test function `f(phi,v)`, the infinitesimal generator and corresponding density equation are

\[
\mathcal Lf=vf_\phi+\frac{\Delta-v-ab\sin\phi}{a}f_v
+\frac{\alpha^2}{2a^2}f_{vv}, \tag{7}
\]

\[
\partial_t p=-\partial_\phi(vp)
-\partial_v\!\left[\frac{\Delta-v-ab\sin\phi}{a}p\right]
+\frac{\alpha^2}{2a^2}\partial_{vv}p. \tag{8}
\]

Equation (8) makes the factor of one half and the amplitude-squared convention explicit.

## 3. Units and nondimensional parameters

No resistor/capacitor values, physical sample clock or measured oscillator calibration are supplied. Simulation outputs therefore use **model time units**, phase in radians, and velocity in radians per model time unit. A numerical `T=50` is not claimed to mean 50 physical seconds.

If (1) is interpreted in physical time measured in seconds and radians are dimensionless, dimensional consistency gives `[a]=s`, `[b]=s^-2`, `[Delta]=s^-1` and `[alpha]=s^-1/2` under (2). The force-noise coefficient in the velocity SDE has units `s^-3/2`.

For reference, let `tau=sqrt(b)t`, `u=v/sqrt(b)` and `B_tau=b^(1/4)W_(tau/sqrt(b))`. Brownian scaling gives a standard Brownian `B`. The same equation becomes

\[
d\phi=u\,d\tau,\qquad
du=(\delta-\gamma u-\sin\phi)d\tau-\beta\,dB_\tau, \tag{9}
\]

where

\[
\delta=\frac{\Delta}{ab},\quad
\gamma=\frac{1}{a\sqrt b},\quad
\beta=\frac{\alpha}{a b^{3/4}}. \tag{10}
\]

Normalized detuning alone does not determine every property of the dynamics: damping and normalized noise also matter. Varying `a` or `b` while keeping Delta and alpha fixed changes several dimensionless groups. E7 is therefore a mechanistic parameter study rather than an isolated effect at fixed normalized damping and noise.

## 4. Deterministic equilibria

Set alpha=0. At an equilibrium, `v=0` and

\[
ab\sin\phi=\Delta.
\]

Define `K=ab>0` and `delta=Delta/K`. When `|delta|<1`, let `theta=arcsin(delta)` in `(-pi/2,pi/2)`. The two branches in every `2pi` period are

\[
(\phi_{s,k},v_s)=(\theta+2k\pi,0),\qquad
(\phi_{u,k},v_u)=(\pi-\theta+2k\pi,0),\quad k\in\mathbb Z. \tag{11}
\]

The subscripts anticipate the stability calculation below. All copies describe the same phase modulo `2pi`, but different copies matter for unwrapped phase and cycle-slip accounting.

If `|delta|>1`, no equilibrium exists because sine cannot exceed one in magnitude. An initial condition of zero is permissible, but it is not a stable equilibrium. At `|delta|=1`, the two branches collide; the resulting critical equilibrium is nonhyperbolic. The project does not classify it as an ordinary stable locked state.

## 5. Linear stability and damping

Write `phi=phi_*+x`, `v=y`. The first-order Taylor expansion is

\[
\frac{d}{dt}\begin{pmatrix}x\\y\end{pmatrix}
=A_*\begin{pmatrix}x\\y\end{pmatrix},\quad
A_*=\begin{pmatrix}0&1\\-b\cos\phi_*&-1/a\end{pmatrix}. \tag{12}
\]

The characteristic equation and eigenvalues are

\[
\lambda^2+\frac1a\lambda+b\cos\phi_*=0,\qquad
\lambda_\pm=\frac{-1\pm\sqrt{1-4a^2b\cos\phi_*}}{2a}. \tag{13}
\]

For the stable branch, `cos(phi_s)=sqrt(1-delta^2)>0`: the trace is negative and determinant positive, so both eigenvalues have negative real part. A positive discriminant gives a stable node, a negative discriminant gives a stable focus with damped oscillation, and a zero discriminant is the repeated-root boundary. Thus `|Delta|<ab` establishes **local asymptotic attraction** of this branch.

For the other branch, `cos(phi_u)=-sqrt(1-delta^2)<0`. Its determinant is negative, hence one eigenvalue is positive and the other negative. It is a saddle. Its one-dimensional stable manifold helps organize basin boundaries in the phase plane.

At the threshold `|delta|=1`, equation (13) gives eigenvalues `0` and `-1/a`; linearization alone cannot supply asymptotic stability. As detuning approaches the threshold from inside, restoring stiffness decreases and the slow eigenvalue approaches zero. This critical slowing and the declining barrier motivate near-threshold experiments, without determining their finite-noise results in advance.

## 6. Tilted potential and energy

Define the tilted potential and mechanical energy

\[
U(\phi)=-\Delta\phi-K\cos\phi,\qquad
H(\phi,v)=\tfrac12av^2+U(\phi). \tag{14}
\]

Since `U'=K sin(phi)-Delta`, the deterministic equation is `a v'=-v-U'`. Differentiate H along a deterministic trajectory:

\[
\frac{dH}{dt}=avv'+U'\phi'=-v^2\le0. \tag{15}
\]

Stable equilibria are local minima of U and saddles are its local maxima in phase. The velocity term is kinetic energy; dissipation decreases H. For nonzero Delta, U is tilted and is not periodic on the real line: `U(phi+2pi)=U(phi)-2pi Delta`. Thus H is not a globally confining Lyapunov function on the unwrapped state space. Energy dissipation does not imply convergence to a single well for every initial condition. Local equilibrium existence, global capture, pull-in range, and sustained running are different questions in a second-order loop. Finite selected trajectories cannot prove a global capture theorem. The warning about hidden PLL attractors is supported in a different PLL setting by [Bianchi et al.](https://arxiv.org/abs/1506.02484); it is not evidence that a hidden attractor has been demonstrated in the present parameter set.

The adjacent right saddle to stable phase theta is `pi-theta`; the left is `-pi-theta`. Direct substitution into (14) gives

\[
\Delta U_+=2K\left[\sqrt{1-\delta^2}-\delta\arccos\delta\right], \tag{16}
\]

\[
\Delta U_-=2K\left[\sqrt{1-\delta^2}+\delta(\pi-\arccos\delta)\right]. \tag{17}
\]

For positive detuning, the right barrier is lower; the difference is `Delta U_- - Delta U_+=2pi Delta`. Negative detuning exchanges the preferred direction. The downhill barrier can be written symmetrically as

\[
\Delta U_{\mathrm{down}}=2K\left[\sqrt{1-\delta^2}-|\delta|\arccos|\delta|\right]
\sim\frac{4\sqrt2}{3}K(1-|\delta|)^{3/2}. \tag{18}
\]

These are deterministic potential heights, not exact stochastic escape rates. Barrier-crossing theory motivates an exponentially sensitive escape mechanism at small noise. A quantitatively valid prefactor and waiting-time approximation depend on damping, noise convention, basin geometry and asymptotic regime. We do not fit a universal Kramers law from a few observed slips. [Kramers' original article](https://doi.org/10.1016/S0031-8914(40)90098-2) supplies the historical escape framework; [Bobrovsky and Schuss](https://epubs.siam.org/doi/10.1137/0142014) specifically formulate second-order PLL slips as an exit problem.

Applying Itô's formula to H in (4) gives the exact identity

\[
dH=\left(-v^2+\frac{\alpha^2}{2a}\right)dt-\alpha v\,dW_t. \tag{19}
\]

The noise injects energy in expectation, balancing dissipation locally or statistically rather than monotonically decreasing H on every sample path.

## 7. Local stochastic approximation and its limits

Near a stable equilibrium, retain linear terms:

\[
dx=y\,dt,\qquad
dy=\left(-b\cos\phi_s\,x-y/a\right)dt-(\alpha/a)dW_t. \tag{20}
\]

This stable linear system is an Ornstein–Uhlenbeck model. If it is allowed to reach its stationary Gaussian law, its covariance solves `AC+CA^T+GG^T=0`, yielding

\[
\operatorname{Cov}(x,y)=0,\qquad
\operatorname{Var}(y)=\frac{\alpha^2}{2a},\qquad
\operatorname{Var}(x)=\frac{\alpha^2}{2ab\cos\phi_s}. \tag{21}
\]

One obtains these identities by setting the derivatives of `E[x^2]`, `E[xy]`, and `E[y^2]` to zero. They provide a local small-fluctuation check and explain why phase variability may increase near the threshold. They are not the stationary variance of the nonlinear unwrapped phase after repeated slips. As the barrier vanishes, the approximation becomes unreliable and the formal variance divergence signals its own breakdown.

For alpha>0 the deterministic fixed point is not an absorbing stochastic state: the velocity diffusion remains nonzero there. “Locked” in the experiments therefore means a specified finite-time retention/capture outcome, not convergence almost surely to an exact point. At Delta=0 a Gibbs density on the cylinder `(phi mod 2pi,v)` is proportional to `exp[-2H/alpha^2]`; the same density repeated across all real phase wells is not normalizable on the unwrapped line. For nonzero detuning the tilted potential cannot define a normalizable equilibrium Gibbs law on the real line; a stationary density with current on the cylinder is a different object.

## 8. Euler–Maruyama and Milstein

For `h=dt`, evaluate F at the old state and use the Gaussian increment (3):

\[
\phi_{n+1}=\phi_n+v_nh, \tag{22}
\]

\[
v_{n+1}=v_n+\frac{\Delta-v_n-ab\sin\phi_n}{a}h
-\frac{\alpha}{a}\Delta W_n. \tag{23}
\]

This is the original program's recurrence. Updating phase with `v_(n+1)` instead would define a different semi-implicit method, so exact replay uses the old velocity.

For a scalar driving Wiener process, the vector Milstein correction is

\[
\tfrac12(DG\,G)\left[(\Delta W_n)^2-h\right]. \tag{24}
\]

Here `DG=0` because G is constant. The correction vanishes in both coordinates: the standard Milstein and Euler–Maruyama updates are exactly identical for the same parameters, initial state and increments. There is no accuracy improvement from relabeling this recurrence “Milstein”. Higher-order methods would need additional drift/integrated-noise terms; the vanishing term alone does not make (22)–(23) exact.

General SDE Euler–Maruyama is commonly introduced with strong order one half, but smooth additive-noise systems admit stronger behavior under suitable assumptions. The present experiment estimates the actual resolved rate using coupled paths; it does not impose either textbook generic order or a fitted value unsupported by the errors. [Higham](https://epubs.siam.org/doi/full/10.1137/S0036144500378302) explains EM/Milstein, strong versus weak error and numerical stability.

## 9. Numerical validation and coupled increments

At alpha=0, compare the same initial-value problem against a tightly toleranced SciPy ODE solution. Since explicit Euler remains first order even when the ODE reference is high order, its error should be reported rather than silently attributed to the reference. Check the actual effective horizon and avoid comparing different endpoint times.

For stochastic mesh comparisons, generate the finest increments once for each independent path. If a coarse step groups `m` fine intervals, use

\[
\Delta W^{(h)}_n=\sum_{j=0}^{m-1}\Delta W^{(h/m)}_{mn+j}. \tag{25}
\]

The sum is Gaussian with variance h and follows the same Brownian realization. Independent random increments on each mesh measure both numerical error and different forcing, so they are unsuitable for a strong-error comparison. An endpoint RMS state error against a finer numerical reference is meaningful only with that reference limitation stated. Weak phase and phase-square differences should be paired per trajectory; a noisy weak-error estimate need not support a regression slope.

## 10. Completed slips and finite-time locking

The unwrapped phase records changes by multiples of `2pi`; wrapped phase alone discards this information. A saddle-phase crossing is useful evidence of motion between wells, but the true deterministic basin boundary is a curve in `(phi,v)`, not the vertical line `phi=phi_u`. The saddle's stable eigenvector gives a local tangent to its stable manifold; backward integration can draw an approximation, with its numerical limitations retained.

The preregistered primary detector records entry into a core of an adjacent stable well followed by continuous residence for 0.5 model time units; the radius is pi/4. Its well index labels the stable centers `phi_s+2pi k`. A committed transition updates the occupied well, and a multiwell jump contributes the appropriate number of completed `2pi` transitions. This operational rule rejects brief recrossings, but may miss rapid running excursions that never settle, and delays the event timestamp by the required residence time. Alternative core sizes and dwell times quantify detector sensitivity. It is not presented as an exact stochastic separatrix or as the unique engineering cycle-slip definition.

At `|delta|>=1`, stable well cores do not exist; equilibrium-based slips and locking probabilities are not applicable. Report rotation via mean unwrapped velocity or net phase advance instead.

The terminal capture rule combines phase proximity and a small tail-mean velocity. Retained lock additionally requires no committed slip. These are finite-horizon operational outcomes, sensitive to the stated cutoffs and initialization. They cannot establish perpetual lock, exact basin membership, or an infinite-time no-slip probability.

A path with no observed first slip by T is right-censored. Its waiting time is at least T, not equal to T. Kaplan–Meier survival and restricted mean waiting time to T retain this information. A conditional mean among observed slips excludes censored paths and must not be called the unconditional mean first-slip time. Zero observed slips can still be consistent with a nonzero event probability; report an independent-path upper bound and label any Poisson exposure-rate approximation.

## 11. What an oscillator/LLM extension can and cannot inherit

A stochastic oscillator network can introduce phases, coupling and correlated Brownian noise explicitly, for example

\[
d\theta_i=\left[\omega_i+K_c\sum_j w_{ij}\sin(\theta_j-\theta_i)\right]dt
+\sigma\,dB_i,\quad
R=\left|N^{-1}\sum_j e^{i\theta_j}\right|. \tag{26}
\]

`R` measures phase concentration: it equals one for aligned phases and decreases with cancellation. Small networks can have appreciable R by chance; high R does not establish correct answers or equal frequencies. This first-order network equation is a separate synthetic model, not a derivation of (4). Topology, weights, delay/dropout implementation and the covariance of the Brownian vector must be stated.

An LLM response embedding has no intrinsic circular coordinate, angular velocity, time scale or sine coupling. Any proposed mapping to phase must be mathematically defined and tested for added predictive value beyond ordinary disagreement on development data. A physics analogy supplies a hypothesis, not validated LLM dynamics or a novelty claim. The formal local experiments can be completed without any API calls; a paid benchmark experiment requires its own approval and evidence gates.
