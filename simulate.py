"""BB84 Project D: explicit one-qubit trajectories, parameter sweeps, and checks.

Run: python simulate.py
No Qiskit or quantum hardware is required. Outputs are simulated, not experimental.
The public check sample is discarded; remaining bits are NOT a secure final key.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
STATES = np.array([[[1, 0], [0, 1]],
                   [[1, 1], [1, -1]]], dtype=complex)
STATES[1] /= np.sqrt(2)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PAULI = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def measure(states, bases, rng):
    """Born-rule measurement in Z (0) or X (1), returning bits and collapse."""
    rotated = states.copy()
    rotated[bases == 1] = states[bases == 1] @ H.T
    p_one = np.clip(np.abs(rotated[:, 1]) ** 2, 0, 1)
    outcomes = (rng.random(len(states)) < p_one).astype(int)
    return outcomes, STATES[bases, outcomes].copy()


def wilson(errors, count):
    """Approximate 95% binomial interval; valid endpoints even with zero errors."""
    z = 1.959963984540054
    q = errors / count
    d = 1 + z*z/count
    center = (q + z*z/(2*count))/d
    radius = z*np.sqrt(q*(1-q)/count + z*z/(4*count*count))/d
    return max(0., center-radius), min(1., center+radius)


def bb84(n, f, p, rng, sample_fraction=0.25):
    """Independent signals; f=intercept probability; E_p(rho)=(1-p)rho+p I/2.

    Alice prepares actual amplitudes. Eve randomly measures selected signals and
    resends the collapsed state. The depolarizing channel is unraveled into
    I/X/Y/Z with probabilities 1-3p/4, p/4, p/4, p/4, before Bob measures.
    """
    if not (n >= 100 and 0 <= f <= 1 and 0 <= p <= 1
            and 0 < sample_fraction < 1):
        raise ValueError('Invalid signal count or probability')
    bits = rng.integers(0, 2, n)
    alice = rng.integers(0, 2, n)
    bob = rng.integers(0, 2, n)
    states = STATES[alice, bits].copy()
    intercepted = rng.random(n) < f
    eve_bases = rng.integers(0, 2, intercepted.sum())
    _, states[intercepted] = measure(states[intercepted], eve_bases, rng)
    operators = rng.choice(4, n, p=[1-3*p/4, p/4, p/4, p/4])
    states = np.einsum('nij,nj->ni', PAULI[operators], states)
    received, _ = measure(states, bob, rng)
    sifted = np.flatnonzero(alice == bob)
    check_n = max(1, int(len(sifted)*sample_fraction))
    check = rng.choice(sifted, check_n, replace=False)
    error = received != bits
    errors = int(error[check].sum())
    all_errors = int(error[sifted].sum())
    low, high = wilson(errors, check_n)
    z_indices = sifted[alice[sifted] == 0]
    x_indices = sifted[alice[sifted] == 1]
    return dict(n=n, f=float(f), p=float(p), sifted=len(sifted),
                sample_n=check_n, sample_errors=errors, qber=errors/check_n,
                ci_low=low, ci_high=high, sifted_errors=all_errors,
                qber_full=all_errors/len(sifted),
                remaining_n=len(sifted)-check_n,
                remaining_errors=all_errors-errors,
                z_n=len(z_indices), z_errors=int(error[z_indices].sum()),
                x_n=len(x_indices), x_errors=int(error[x_indices].sum()))


def expected_qber(f, p):
    return f/4 + p/2 - f*p/4


def aggregate(rows):
    n = sum(r['sample_n'] for r in rows)
    k = sum(r['sample_errors'] for r in rows)
    low, high = wilson(k, n)
    z_n = sum(r['z_n'] for r in rows)
    x_n = sum(r['x_n'] for r in rows)
    return dict(f=rows[0]['f'], p=rows[0]['p'], trials=len(rows),
                signals=sum(r['n'] for r in rows), sample_n=n, sample_errors=k,
                qber=k/n, ci_low=low, ci_high=high,
                theory=expected_qber(rows[0]['f'], rows[0]['p']),
                sifted=sum(r['sifted'] for r in rows),
                mean_sift_fraction=np.mean([r['sifted']/r['n'] for r in rows]),
                trial_sd=float(np.std([r['qber'] for r in rows], ddof=1)),
                q_z=sum(r['z_errors'] for r in rows)/z_n,
                q_x=sum(r['x_errors'] for r in rows)/x_n)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_checks():
    """Independent physical and statistical checks, not just code-path tests."""
    # Pauli trajectory ensemble equals the declared density-matrix channel.
    for state in STATES.reshape(-1, 2):
        rho = np.outer(state, state.conj())
        for p in [0., .2, 1.]:
            mixed = sum(w * u @ rho @ u.conj().T for w, u in
                        zip([1-3*p/4, p/4, p/4, p/4], PAULI))
            assert np.allclose(mixed, (1-p)*rho + p*np.eye(2)/2)
    # Same basis must reproduce every prepared bit; other basis must be fair.
    rng = np.random.default_rng(771)
    for basis in [0, 1]:
        for bit in [0, 1]:
            states = np.tile(STATES[basis, bit], (40000, 1))
            same, _ = measure(states, np.full(40000, basis), rng)
            other, _ = measure(states, np.full(40000, 1-basis), rng)
            assert np.all(same == bit)
            assert abs(other.mean()-.5) < .015
    for f, p in [(0, 0), (1, 0), (0, 1), (.4, .2)]:
        r = bb84(200000, f, p, rng)
        q = expected_qber(f, p)
        se = np.sqrt(max(q*(1-q), 1e-8)/r['sample_n'])
        assert abs(r['qber']-q) < 6*se + .001
        assert abs(r['sifted']/r['n']-.5) < .01
        assert r['remaining_n'] + r['sample_n'] == r['sifted']
        if f == p == 0:
            assert r['sifted_errors'] == 0
    return 'PASS: Born rule, collapse, channel identity, sifting, and QBER limits'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--signals', type=int, default=20000)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error('--trials must be at least 2')
    start = time.perf_counter()
    checks = self_checks()
    rng = np.random.default_rng(args.seed)
    raw, sweep = [], []
    for mode in ['eve', 'noise']:
        for strength in np.linspace(0, 1, 11):
            f, p = (strength, 0.) if mode == 'eve' else (0., strength)
            rows = [bb84(args.signals, f, p, rng) for _ in range(args.trials)]
            raw.extend(dict(mode=mode, trial=i, **r) for i, r in enumerate(rows))
            sweep.append(dict(mode=mode, **aggregate(rows)))
    # Matched QBER at several strengths, using the mapping p=f/2.
    pairs = []
    for f in [.2, .4, .8, 1.]:
        ar = [bb84(args.signals, f, 0., rng) for _ in range(args.trials)]
        br = [bb84(args.signals, 0., f/2, rng) for _ in range(args.trials)]
        raw.extend(dict(mode='matched_eve', trial=i, **r) for i, r in enumerate(ar))
        raw.extend(dict(mode='matched_noise', trial=i, **r) for i, r in enumerate(br))
        a, b = aggregate(ar), aggregate(br)
        delta = a['qber']-b['qber']
        se = np.sqrt(a['qber']*(1-a['qber'])/a['sample_n'] +
                     b['qber']*(1-b['qber'])/b['sample_n'])
        pairs.append(dict(f=f, p=f/2, eve_q=a['qber'], noise_q=b['qber'],
                          difference=delta, diff_low=delta-1.96*se,
                          diff_high=delta+1.96*se, eve_sample_n=a['sample_n'],
                          noise_sample_n=b['sample_n'], eve_q_z=a['q_z'],
                          eve_q_x=a['q_x'], noise_q_z=b['q_z'], noise_q_x=b['q_x']))
    # Additional mixed-cause experiment checks the composition formula.
    mr = [bb84(args.signals, .4, .2, rng) for _ in range(args.trials)]
    raw.extend(dict(mode='mixed', trial=i, **r) for i, r in enumerate(mr))
    mixed = aggregate(mr)
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    save_csv(results/'trials.csv', raw)
    save_csv(results/'sweep.csv', sweep)
    save_csv(results/'matched.csv', pairs)
    # Root of 1-2 h2(q), solved without a SciPy dependency.
    lo, hi = 0., .5
    for _ in range(70):
        q = (lo+hi)/2
        h = -q*np.log2(q)-(1-q)*np.log2(1-q)
        if 1-2*h > 0:
            lo = q
        else:
            hi = q
    elapsed = time.perf_counter()-start
    summary = dict(seed=args.seed, signals_per_trial=args.signals,
                   trials_per_setting=args.trials, sample_fraction=.25,
                   python=platform.python_version(), numpy=np.__version__,
                   elapsed_seconds=elapsed, checks=checks,
                   security_threshold=(lo+hi)/2, sweep=sweep, matched=pairs,
                   mixed=mixed,
                   protocol_signals=sum(r['signals'] for r in sweep) +
                   len(pairs)*2*args.signals*args.trials + mixed['signals'])
    (results/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(checks)
    print(f'Experiments: {summary["protocol_signals"]:,} signals in {elapsed:.2f} s')
    print('Matched settings:', json.dumps(pairs, indent=2))
    print('Combined causes:', mixed['qber'], 'theory:', mixed['theory'])


if __name__ == '__main__':
    main()
