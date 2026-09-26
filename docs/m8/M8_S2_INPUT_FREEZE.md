# M8 P1-S2 CPU input-only freeze

This record binds the [frozen S2 scientific protocol](M8_S2_PROTOCOL.md) to a detector-free
five-feature input implementation and a complete input-only ledger. The scientific protocol is
unchanged. No B2/C2/D2/F2 detector output, S2 repeatability result, or full-corpus inference
result is created by this work.

## Chronology and authoritative inputs

The input-construction implementation was committed as
`bf098b319744f1ec1df08207c1cd93853b1f31ae` **before** real-corpus generation. The
[compact manifest](../../benchmarks/m8/inputs/m8_s2_input_manifest.json),
[CPU input characterization](../../benchmarks/m8/diagnostics/m8_s2_input_characterization.json),
and [machine-readable freeze record](../../benchmarks/m8/preregistration/m8_s2_input_freeze.json)
bind that commit. The detailed JSONL ledger is private, Drive-backed project state; only its
logical filename, size, SHA256, and record count are tracked. No KITTI points or reconstructed
point matrices are committed.

| Evidence | Frozen identity |
|---|---|
| Full private ledger | `m8_s2_input_ledger.jsonl`; 7,728,782 bytes; SHA256 `a3ed54b276f77fb784035045b079573cd4e4ddfedc9d0f8eb774c1340a59396b`; 1,712 records |
| Compact tracked manifest | SHA256 `239b563d5f850f2f20809950eca9d56f1a700677766afce8489099e940554ecc` |
| CPU input characterization | SHA256 `c327dafe6287335c971fa8c5c5f2a15680c08aafcc367e4cdccda4e6e39f86db` |
| Frozen M7 compact manifest | SHA256 `8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90` |

Before construction, the M8 accepted source-input ledger, full M6 transform ledger, M6 result
asset, and frozen M7 compact manifest passed their exact recorded size/hash checks. The unchanged
M7 source adapter and the M8 frozen five-feature adapter independently reconstructed each of the
428 A/H10, E/H5, A2/H10, and E2/H5 source pairs. The M8 full XYZIT commitments and their M7
XYZT projections were exact before an S2 arm was constructed.

## Input-only gates

In canonical frame-major order, the generator constructed B2, C2, D2, and F2 by lifting the
corresponding unchanged M7 intervention result into the verified M8 A2 source. It checked each
projected XYZT SHA256 and selected-row SHA256 against the frozen M7 compact manifest with zero
tolerance. Raw intensity came only from the selected M8 A2 source rows.

The pairwise structural gates use `CPU_ANALYTIC_STRUCTURAL_IDENTITY`:

- B2/A2: identical selected rows, XYZ bytes, raw intensity bytes, and CPU analytic candidate
  coordinate arrays/hashes.
- C2: M7 selected A2 rows, exact E2 total point count, all current rows retained, historical
  ordered subset, and native intensity/lag.
- D2/C2: identical selected rows, XYZ/intensity bytes, point count, and CPU analytic candidate
  coordinate arrays/hashes; only lag changes.
- F2: M7 complete ranks 0,2,4,6,8,10 with native A2 XYZ, intensity, lag, and global order.

All four arms passed the exact M7 XYZT projection and selected-row gates: B2 **428/428**, C2
**428/428**, D2 **428/428**, F2 **428/428**, for **1,712/1,712** total. The B2/A2 row, XYZ,
intensity, and CPU analytic coordinate identities passed **428/428** each. C2 matched the E2
point count and exact selected A2 values in **428/428**. The D2/C2 row, XYZ, intensity, count,
and CPU analytic coordinate identities passed **428/428** each. F2 had the frozen complete ranks
and exact A2 subset values in **428/428**. A separate fresh CPU process reconstructed all seven
frozen sentinel frames through new adapters and compared every detailed ledger field for their
28 conditions: **28/28 exact**. No detector calls occurred.

The detailed ledger records each condition's source commitments, full XYZIT and projected XYZT
hashes, standalone XYZ/intensity hashes, selected-row identity, point count, sweep membership,
per-sweep counts, M7 B/D lag-scale bits, M7 C/D integer quotas and seeds, F2 ranks, and CPU
analytic structural coordinate identities. It is streamed one frame at a time and contains no
detector predictions or ground-truth outcomes.

The [CPU characterization](../../benchmarks/m8/diagnostics/m8_s2_input_characterization.json)
publishes per-arm minimum, median, mean, and maximum for point counts, point-count ratios to A2
and E2, lag support and span, CPU analytic candidate-pillar counts, and per-sweep counts. B2's
lag-scale range was 0.499579–0.500226. C2 retained 53.69%–55.55% of A2 points, with no
zero-quota frames. F2 had more points than E2 in 115/428 frames and a longer lag span in
428/428; its complete ranks include both the current and oldest A2 sweep, preserving full A2
temporal support. These are input-structure observations, not detector outcomes.

## Structural arithmetic and future runtime handoff

For the first H10 frame, the source input identity matched the published historical census. The
CPU analytic helper counted 30,623 candidate pillars, while that census recorded 30,624 under
CUDA coordinate arithmetic. The CPU helper documents possible near-cell-boundary float32
arithmetic differences. The exact one-pillar cause was not localized here. CPU/CUDA count
equality is not an input-freeze requirement; neither count is called wrong. The S2 pairwise
CPU-coordinate gates compare inputs under the *same* CPU arithmetic definition.

Before future detector inference on an explicitly selected external runtime, qualification must
reconstruct all 28 frozen S2 sentinel inputs with exact XYZIT hashes, compute exact CUDA candidate
pillar counts and coordinate identities, and require same-runtime CUDA B2/A2 and D2/C2 structural
identity. That work is not part of this freeze; no runtime is bound and inference remains
unauthorized.
