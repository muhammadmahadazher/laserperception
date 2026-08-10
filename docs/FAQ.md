# Frequently asked questions

## What is LaserPerception?

An open-source research framework for semantic transfer across heterogeneous 3D LiDAR point clouds.

## What problem does it study?

Zero-shot semantic transfer from vehicle-mounted SemanticKITTI LiDAR to airborne DALES LiDAR under
a shared six-class geometry-only ontology.

## Is it an object detector or a 2D LiDAR framework?

No. Current scope is 3D point-wise semantic segmentation.

## Why SemanticKITTI and DALES?

Their automotive and airborne views provide a concrete cross-view transfer problem.

## Why xyz-only?

Shared geometry avoids relying on sensor attributes that are unavailable or incomparable across
domains. Loaders still preserve remission and LAS attributes.

## Are raw LAS coordinates normalized while loading?

No. `min_xyz` is a separate transform that returns a new cloud and records its translation.

## Is LaserPerception production-ready?

No. It is an early research preview with no trained model or measured benchmark.

## Can companies use it commercially?

Original code is Apache-2.0 licensed. Dataset and third-party terms remain separate.

## Are datasets included? Does the core require a GPU?

No datasets are included, and the current package/tests are CPU-only.

## What follows Experiment 001?

Subject to evidence from bounded real-dataset audits: deliberate sparse-backend selection, a minimal
baseline, source training, zero-shot evaluation, and one evidence-driven ablation.
