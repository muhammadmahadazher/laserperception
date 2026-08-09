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

Commit reusable code and small configuration, not datasets, checkpoints, logs, or generated output.
Store artifacts externally with hashes or immutable references where appropriate.

Unmeasured values are `Pending measurement`. One seeded run must not be presented as evidence of
variance. GPU measurements must state device scope and whether allocated or reserved memory was
sampled.
