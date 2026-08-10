# Changelog

All notable changes to LaserPerception will be documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use semantic versioning once
releases begin.

## [Unreleased]

### Added

- Professional open-source repository foundation and CPU CI.
- Canonical validated `PointCloud` representation.
- KITTI/SemanticKITTI scan I/O and packed label decoding.
- LAS/optional LAZ loading with metadata and attribute preservation.
- Explicit `min_xyz` coordinate normalization.
- Verified six-class SemanticKITTI and DALES ontology mappings.
- Experiment 001 configuration, synthetic tests, and research documentation, including a pinned
  CVGC reference investigation.
- Directory-level SemanticKITTI adapter with pinned official splits and scan/label validation.
- Memory-conscious DALES adapter with selective chunked reading and deterministic grid patches.
- CPU-only dataset audit CLI with redacted JSON reports and ontology-coverage statistics.
- Synthetic adapter, boundary, conservation, memory-path, and audit regression tests.
