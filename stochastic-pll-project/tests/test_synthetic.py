"""Synthetic network checks isolate noise, communication and timing semantics."""
import numpy as np
import pytest
from src.synthetic_agents import (SyntheticConfig, topology_matrix, noise_covariance,
    coupled_drift, delayed_index, simulate_networks, summarize_networks)


def test_topologies_are_connected_symmetric_and_degree_normalized():
    for n in (3, 8, 20):
        for name in ('complete', 'ring', 'star', 'sparse'):
            a = topology_matrix(n, name)
            assert np.array_equal(a, a.T)
            assert not np.any(np.diag(a))
            assert np.all(a.sum(1) > 0)
            reach = np.linalg.matrix_power(a + np.eye(n), n - 1)
            assert np.all(reach > 0)
    assert np.array_equal(topology_matrix(3, 'ring'), topology_matrix(3, 'complete'))


def test_noise_covariance_matches_equicorrelated_construction():
    rng = np.random.default_rng(31)
    z, common = rng.normal(size=(120000, 3)), rng.normal(size=(120000, 1))
    for rho in (0, .5, 1):
        sample = np.sqrt(1-rho)*z + np.sqrt(rho)*common
        assert np.allclose(np.cov(sample.T), noise_covariance(3, rho), atol=.014)
    assert np.array_equal(sample[:, 0], sample[:, 2])


def test_dropout_does_not_renormalize_remaining_neighbors():
    a = topology_matrix(3, 'complete')
    phase = np.array([0., np.pi/2, np.pi/2])
    omega = np.zeros(3)
    full = coupled_drift(phase, omega, a, 2.)
    mask = np.ones_like(a); mask[0, 1] = 0
    partial = coupled_drift(phase, omega, a, 2., mask)
    assert full[0] == pytest.approx(2.)
    assert partial[0] == pytest.approx(1.)
    assert np.allclose(coupled_drift(phase, omega, a, 2., np.zeros_like(a)), omega)


def test_delay_uses_previous_relative_errors_and_constant_prehistory():
    assert delayed_index(0, 20) == 0
    assert delayed_index(25, 20) == 5
    a = topology_matrix(3, 'complete')
    phases = np.array([0., .5, 1.])
    old = np.array([0., -.3, -.8])
    assert not np.allclose(coupled_drift(phases, np.zeros(3), a, 2.),
                           coupled_drift(old, np.zeros(3), a, 2.))


def test_local_error_is_first_order_for_deterministic_integrator(tmp_path):
    a = topology_matrix(3, 'complete')
    init = np.array([[0., .7, -.3]])
    common = dict(n_agents=3, topology='complete', rho=0., dropout=0., delay=0.,
                  n_trajectories=1, T=.4, sigma=0., coupling=2., sample_dt=.01)
    from scipy.integrate import solve_ivp
    frequencies = np.array([-.5, 0, .5])
    ref = solve_ivp(lambda t, x: coupled_drift(x, frequencies, a, 2.),
                    (0., .4), init[0], rtol=1e-12, atol=1e-13).y[:, -1]
    errors = []
    for dt in (.01, .005, .0025):
        out = simulate_networks(SyntheticConfig(**common, dt=dt),
                                tmp_path, initial_phases=init, frequencies=frequencies)
        errors.append(np.linalg.norm(out['phase_samples'][0, -1]-ref))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_shared_noise_cancels_from_relative_dynamics_even_with_delay(tmp_path):
    common = dict(n_agents=3, topology='star', rho=1., dropout=.3, delay=.2,
                  n_trajectories=8, T=1., dt=.01, sample_dt=.1)
    noisy = simulate_networks(SyntheticConfig(**common, sigma=.8), tmp_path)
    quiet = simulate_networks(SyntheticConfig(**common, sigma=0.), tmp_path)
    assert np.array_equal(noisy['order_parameter'], quiet['order_parameter'])
    for metric in ('tail_R_mean', 'tail_pairwise_disagreement', 'tail_frequency_dispersion'):
        assert np.array_equal(noisy['rows'][metric], quiet['rows'][metric])
    assert not np.allclose(noisy['phase_samples'], quiet['phase_samples'])


def test_reproducibility_finite_outputs_and_network_level_intervals(tmp_path):
    c = SyntheticConfig(n_agents=8, topology='sparse', rho=.5, dropout=.3, delay=.2,
                        n_trajectories=12, T=1., dt=.01, sample_dt=.1)
    first = simulate_networks(c, tmp_path)
    second = simulate_networks(c, tmp_path)
    for name in first['rows'].dtype.names:
        assert np.array_equal(first['rows'][name], second['rows'][name])
    assert np.isfinite(first['order_parameter']).all()
    assert np.all((first['order_parameter'] >= 0) & (first['order_parameter'] <= 1+1e-14))
    summary = summarize_networks(first['rows'], c)
    assert summary['n'] == 12
    assert summary['tail_R_mean_ci_low'] <= summary['tail_R_mean_mean'] <= summary['tail_R_mean_ci_high']
    r = first['rows']['tail_R_mean']
    assert summary['tail_R_mean_sd'] == pytest.approx(np.std(r, ddof=1))
    assert len(np.unique(first['rows']['brownian_seed'])) == 12


def test_invalid_parameters_fail_closed():
    for kwargs in ({'rho': -1}, {'rho': 1.1}, {'dropout': 1.1}, {'delay': -.1},
                   {'dt': .03, 'delay': .2}, {'n_agents': 1}, {'n_trajectories': 0}):
        with pytest.raises(ValueError):
            SyntheticConfig(**kwargs).validate()


def test_native_noise_has_correct_network_mean_variance(tmp_path):
    c = SyntheticConfig(n_agents=3, n_trajectories=10000, T=.2, dt=.01,
                        sample_dt=.1, coupling=0., sigma=.4, rho=.5)
    data = simulate_networks(c, tmp_path, initial_phases=np.zeros((10000, 3)),
                             frequencies=np.zeros(3))
    expected = c.sigma**2*c.T*(c.rho+(1-c.rho)/c.n_agents)
    assert np.var(data['rows']['terminal_mean_phase'], ddof=1) == pytest.approx(expected, rel=.05)


def test_total_dropout_removes_all_coupling_and_native_delay_matches_reference(tmp_path):
    init = np.array([[0., .7, -.3]])
    frequencies = np.array([-.5, 0, .5])
    base = dict(n_agents=3,n_trajectories=1,T=.4,dt=.01,sample_dt=.01,sigma=0.)
    dropped = simulate_networks(SyntheticConfig(**base,dropout=1.),tmp_path,
                                initial_phases=init,frequencies=frequencies)
    uncoupled = simulate_networks(SyntheticConfig(**base,coupling=0.),tmp_path,
                                  initial_phases=init,frequencies=frequencies)
    assert np.array_equal(dropped['order_parameter'],uncoupled['order_parameter'])
    delayed = simulate_networks(SyntheticConfig(**base,delay=.2),tmp_path,
                                initial_phases=init,frequencies=frequencies)
    reference = [init[0].copy()]
    for step in range(40):
        past = reference[delayed_index(step,20)]
        reference.append(reference[-1]+.01*coupled_drift(past,frequencies,topology_matrix(3,'complete'),2.))
    assert np.allclose(delayed['phase_samples'][0],reference,atol=1e-14,rtol=0)
    clean = simulate_networks(SyntheticConfig(**base),tmp_path,
                              initial_phases=init,frequencies=frequencies)
    assert not np.allclose(delayed['order_parameter'],clean['order_parameter'])
