# Project D: Is It an Eavesdropper or Just a Bad Channel?

Completed BB84 simulation study for the supplied Quantum Technologies mini-project.

## Run

Use Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python simulate.py
python build_report.py
```

The default seed is 20261019. Each setting uses 20 independent trials with 20,000
signals each. There are 620 experimental trials (12.4 million signals), plus
separate physical/statistical validation runs. This is a local simulation and
requires no Qiskit, cloud access, credentials, or quantum hardware.

## Files

- `Project_D_Report.pdf`: 10-page report covering all five project stages.
- `Project_D_Report.md`: editable report text with relative figure links.
- `simulate.py`: commented single-qubit state simulation, checks, and experiments.
- `build_report.py`: reproducible figure/report builder using existing results.
- `results/trials.csv`: all 620 trial records, including matched and mixed runs.
- `results/sweep.csv`: pooled interception and channel-noise sweeps.
- `results/matched.csv`: independent matched-QBER comparisons and difference intervals.
- `results/summary.json`: numerical results, seed, software versions, and timing.
- `figures/`: four figures in PNG and vector PDF formats.

## Model

Alice and Bob independently choose uniformly random Z/X bases. Alice encodes a
random bit in an explicit complex two-amplitude state. Eve intercepts each signal
independently with probability f, measures in a random basis, and resends the
collapsed state. The channel is E_p(rho)=(1-p)rho+p I/2. Its Pauli trajectory
probabilities are I: 1-3p/4 and X,Y,Z: p/4 each. **p is not the probability of a
bit flip or the total probability of applying a nonidentity Pauli.**

Bob measures with the Born rule. After basis sifting, a random 25% check sample is
disclosed and discarded. Primary QBER results use this sample. Full-sifted and
remaining-string errors are simulator diagnostics, not public protocol outputs.
Basis-specific rates in matched.csv use full-sifted diagnostic data.

Predictions: q_E=f/4, q_N=p/2, and combined q=f/4+p/2-fp/4. Setting p=f/2 makes
interception-only and noise-only QBER equal. The 95% Wilson intervals and matched
normal difference intervals quantify simulation uncertainty, not composable
finite-key security. The approximately 11% threshold is an ideal asymptotic
single-photon, symmetric-error, one-way post-processing reference.

## Scope and validation

Implemented: random bits/bases, preparation, intercept-resend, depolarization,
measurement, sifting, public error estimation, sweeps, repeated trials, and plots.
The report also discusses photons, threshold assumptions, and multiple hops.
Not implemented: error correction, privacy amplification, optical loss, detector
imperfections, or a classical-message authentication mechanism. The remaining
string is therefore **not a final secret key**. Authentication is assumed.

Self-checks validate same/opposite-basis Born probabilities, the Pauli ensemble
against the density-matrix channel, ideal/full-interception/full-noise QBER,
combined causes, and sifting/count conservation. Assertions stop execution on
failure. All included default-run checks passed.

Optional custom experiments:

```sh
python simulate.py --seed 123 --signals 40000 --trials 30
```

The report narrative describes the default design. Review it if changing these
parameters; some design counts and example settings are intentionally fixed.
The precise generating Python and NumPy versions are in summary.json. Package
ranges support installation but do not promise bit-identical results across
every future library release. This package contains no hardware measurements.

## Before submission

Read and understand the model, add your name/group details if required, and follow
your course's rules for acknowledging assistance. The report makes no claim of
physical experimentation or implementation of final secret-key distillation.
