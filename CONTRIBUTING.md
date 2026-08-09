# Contributing

LaserPerception welcomes focused, reproducible contributions. The project is a research preview, so
scientific clarity and small reviewable changes matter more than feature breadth.

## Development setup

Use Python 3.10–3.13 and keep large environments off synchronized storage when practical.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,laz]"
pre-commit install
```

Create branches with descriptive prefixes such as `feat/`, `fix/`, `docs/`, `test/`, or `chore/`.
Use Conventional Commit-style messages when practical.

## Quality checks

Run before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy src
python -m pytest
python -m build
```

Tests must be deterministic, synthetic, and independent of downloaded datasets. Add malformed-input
coverage for file readers and test scientific invariants, not only happy paths.

## Dataset adapters and ontology mappings

An adapter must preserve raw coordinates and available attributes, keep normalization separate,
accept an explicit dataset root, and document supported layout/version. Never commit data fixtures
copied from a public dataset.

Mapping changes require an authoritative source for numeric IDs, a documented grouping rationale,
explicit ignore behavior, and tests. Do not infer IDs from memory or silently change an experiment's
label space.

## Reproducible experiments

Experiment contributions must include an exact configuration and follow
`docs/REPRODUCIBILITY.md`. Do not add illustrative benchmark values. Unmeasured fields say
`Pending measurement`. Do not commit outputs, weights, checkpoints, or logs.

## External implementations and licensing

Prefer implementing small interfaces from documented specifications. Before incorporating external
code, verify its license, preserve required notices, cite its source, describe modifications, and
update `THIRD_PARTY_NOTICES.md`. Dataset terms are separate from Apache-2.0.

## Pull request checklist

- [ ] The change is within current project scope.
- [ ] Public behavior and configuration are documented.
- [ ] Tests cover success and failure behavior without dataset downloads.
- [ ] Ruff, mypy, pytest, and package build pass.
- [ ] No dataset, secret, checkpoint, environment, or generated output is included.
- [ ] External sources and licenses are cited where required.
- [ ] Benchmark values are measured and reproducible, or say `Pending measurement`.
