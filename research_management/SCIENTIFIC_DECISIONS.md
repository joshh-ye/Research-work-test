# Scientific Decisions

Record each decision as: ID · date · decision · rationale · evidence · alternatives considered · decided by.

## SD-001 · 2026-10-19 · Metric definitions
- Decision: report Pearson r, squared Pearson r (r²), and prediction R² (1 − SS_res/SS_tot) as distinct,
  explicitly labelled quantities. R² for pretrained baselines is labelled "validation-calibrated R²".
- Rationale: raw-scale R² of an uncalibrated pretrained track is meaningless (previously observed −193).
- Evidence: `reports/qing_deliverables/Assumptions.md`, `Scientific_Review.md`.
- Decided by: user requirement (setup).

## SD-002 · 2026-10-19 · Reuse Qing's completed analysis
- Decision: treat `reports/qing_deliverables/` as completed; re-verify, do not re-run, unless a new
  question requires it (record the justification here).
- Decided by: user requirement (setup).
