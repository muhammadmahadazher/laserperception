# Reproducibility policy

Every experiment record must include:

- exact Git SHA and dirty state;
- immutable config snapshot;
- dataset versions, splits, terms, and integrity evidence;
- preprocessing, ontology, feature, and metric versions;
- all seeds and deterministic settings;
- Python environment, OS, CPU, RAM, GPU, driver, and CUDA where applicable;
- wall-clock boundaries and peak-memory measurement method; and
- deviations, failures, restarts, and checkpoint-selection rules.

## Dataset audit records

Before training design, use `python -m laserperception.audit` with a bounded sample/tile limit. Retain
the JSON report externally with the experiment record. The report schema records adapter version,
official split name, explicit subset identifiers, ontology name, normalization setting, patch
policy, counts, coordinate ranges, label histograms, ignored fraction, timestamp, and Git commit
when available.

Audit JSON intentionally omits absolute dataset roots. A SemanticKITTI subset is represented by
sequence/frame IDs; DALES uses tile IDs and patch grid coordinates. Reports under `audit-reports/`
are ignored so local real-dataset evidence is not accidentally committed. If a report is deliberately
published later, review its provenance, licensing, privacy, and size first.

The DALES audit must record patch size, tile-min grid origin, half-open boundaries, empty-cell count,
non-finite points, and whether per-patch normalization was requested. An official dataset split and
an experiment subset are separate fields and must never be conflated.

Commit reusable code and small configuration, not datasets, checkpoints, logs, or generated output.
Store artifacts externally with hashes or immutable references where appropriate.

Unmeasured values are `Pending measurement`. One seeded run must not be presented as evidence of
variance. GPU measurements must state device scope and whether allocated or reserved memory was
sampled.
