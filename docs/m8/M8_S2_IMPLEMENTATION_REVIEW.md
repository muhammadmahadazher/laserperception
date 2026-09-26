# M8 P1-S2 CPU-only input implementation review

The [frozen S2 scientific protocol](M8_S2_PROTOCOL.md) governs this input-only implementation.
Its committed Markdown SHA256 is
`218ef2dcc03fa4ff75562e02f368f7624f16a4c768b1758147c1d46ac1d9c53d`.
This implementation does not load ground truth, a detector, Torch, or an accelerator runtime.
It does not authorize S2 repeatability or full-corpus inference.

## Implementation identity and authority

The input-generating implementation commit is
`bf098b319744f1ec1df08207c1cd93853b1f31ae`. The private full ledger and tracked compact
artifacts bind this exact commit. Any later change to source reconstruction, row selection, lag
construction, intensity selection, hashes, structural validation, or ledger contents invalidates
the generated evidence and requires complete regeneration and replay.

The canonical M7 implementation commit is
`c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`. Git comparison before generation found no
intervention or provenance drift in `benchmarks/m7/interventions.py`, `provenance.py`,
`prepare_inputs.py`, or `structural_validation.py`. The M7 source adapter reconstructs the
frozen A/H10 and E/H5 XYZT pairs from the exact M6 transform ledger, while the M8 frozen input
adapter reconstructs the accepted A2/H10 and E2/H5 XYZIT pairs. Before any S2 arm is built,
both M8 full-source hashes and both M7 XYZT projections must match their recorded commitments.

The [S2 input lift](../../src/laserperception/detection/m8_s2_input.py) has no M7 package
dependency in the core wheel. The [CPU generation tool](../../benchmarks/m8/prepare_s2_inputs.py)
calls the unchanged M7 `construct_b`, `construct_c`, `construct_d`, and `construct_f` functions,
then lifts their exact selected global A rows and lag values into the verified five-feature M8
A2 source. It does not implement a second quota, seed, lag, or row-selection algorithm.

| Arm | Five-feature lift | Exact relation |
|---|---|---|
| B2 | Copy every A2 row; replace only lag with M7 B lag | A2 row order, XYZ, and intensity unchanged |
| C2 | Select A2 rows at M7 C `selected_global_rows` | Exact E2 total point count; native A2 values |
| D2 | Copy C2 rows; replace only lag with M7 D lag | C2 rows, XYZ, and intensity unchanged |
| F2 | Select A2 rows at M7 F `selected_global_rows` | Complete ranks 0,2,4,6,8,10; native values |

Every generated condition is projected to little-endian C-contiguous float32 XYZT and checked
against the corresponding frozen M7 condition's `model_ready_sha256` and selected-row SHA256.
Full XYZIT and standalone intensity hashes are recorded separately. A failure stops generation;
there is no tolerance or alternate selection.

## CPU structural identity and cross-runtime caveat

Structural fields in the new ledger use the explicit namespace
`cpu_analytic_candidate_pillar_count` and
`cpu_analytic_candidate_coordinate_sha256`. They come only from the CPU
`candidate_dynamic_pillar_coordinates` function. B2/A2 and D2/C2 are compared within this same
CPU arithmetic definition, including the coordinate arrays themselves; these are valid spatial
invariance gates for transformations that change only lag.

The existing historical H10 census reports **30,624** CUDA candidate pillars for
`2011_09_26_drive_0001/0000000010/H10`, while the CPU analytic helper reports **30,623** for
the matching frozen source input. The helper already documents that CUDA float32 division may
round points very near cell boundaries differently. This task did not establish the exact cause
of the one-pillar difference. Neither count is treated as erroneous, and CPU/CUDA count equality
is not an S2 input-freeze gate. These CPU summaries are not a new GPU runtime binding or a
claim of byte-identical CUDA behavior.

## Future runtime barrier

Before any S2 detector inference on a separately authorized external GPU, runtime qualification
must reconstruct all 28 frozen sentinel B2/C2/D2/F2 inputs with exact XYZIT hashes under that
runtime and compute CUDA candidate pillar counts and coordinate identities separately. It must
check same-runtime CUDA B2/A2 and D2/C2 structural identity, with required reference inputs
reconstructed under the same bound runtime. No CUDA check was performed for this input freeze.
