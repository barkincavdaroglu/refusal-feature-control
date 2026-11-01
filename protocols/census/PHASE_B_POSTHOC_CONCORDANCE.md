# S16 Phase-B post-hoc behavioral concordance

Status: **complete, exploratory, and non-causal; Phase C is not authorized**

This analysis joins the existing Phase-B activation census for feature
`C = L31:F49541` to the blind R/S/P/C labels from S10. It ran on CPU over
existing artifacts only. No model was loaded, no prompt was generated, and no
intervention was performed.

## Basic version

The join is exact: all 180 records matched by intent and frame, and the prompt
text on disk independently reproduced the same SHA-256 stored by the S10
generation record, the S10 prompt record, and the S16 census record. No record
was excluded, and no prompt text appears in the output.

On the 120 harmful records, C was active for 72 refusal-like answers and one
compliance-like answer. When C was silent, 17 answers still refused and 30
complied:

| Final-token C | Refusal-like (R/S) | Compliance-like (P/C) |
|---|---:|---:|
| Active (> 0) | 72 | 1 |
| Silent (= 0) | 17 | 30 |

The refusal-rate difference between active and silent records was `0.625`
(intent-cluster bootstrap 95% interval `[0.427, 0.789]`). C distinguished
refusal-like from compliance-like records with AUC `0.875` overall
(`[0.783, 0.937]`), `0.758` in fictional frames (`[0.694, 0.833]`), and
`0.743` in direct frames. The direct-frame interval was `[0, 1]`, so that
subgroup estimate is highly uncertain.

Across the 60 matched direct/fictional frame pairs, 23 answers flipped from
refusal-like to compliance-like. C was silenced in 22 of those 23 flips
(`95.7%`), but it was also silenced in 17 pairs that did not flip. Put simply:
C being present is strongly associated with refusal, and its disappearance
nearly always accompanies a flip, but its disappearance is not enough to make
the model comply. C is a measured internal update, not the model's complete
refusal state.

All 60 benign S10 records had C silent and were compliance-like. The 24 fresh
reference prompts also had C silent, but have no behavior labels. In the S1
side cells, instruction override left C active on `10/10` harmful prompts
(mean `23.075`; all 10 refusal-like), while unsafe-model-simulation wording
activated it on `10/10` harmful prompts (mean `8.516`; all refusal-like) and
`3/4` benign prompts (mean `4.594`; two refusal-like and two compliance-like).

## Academic version

The independent unit is intent, not frame: 20 harmful and 10 benign intents,
each with six nested frames. With 10,000 NumPy PCG64 resamples at frozen seed
`2522210527409210963`, the harmful active-versus-silent risk difference was
`0.624599` with percentile interval `[0.426740, 0.789474]`. Harmful AUC was
`0.875317` (`[0.783245, 0.936782]`); fictional AUC was `0.757576`
(`[0.693548, 0.833333]`). Direct AUC was `0.743304`, but only 8,746 bootstrap
draws contained both outcome classes and its interval was `[0, 1]`; undefined
class-degenerate draws were omitted and counted in the machine-readable
record.

The paired flip-by-silencing table was `(flip, silenced)=22`,
`(flip, not-silenced)=1`, `(not-flip, silenced)=17`, and
`(not-flip, not-silenced)=20`. Thus
`P(silenced | flip)=0.9565` and `P(flip | silenced)=0.5641`. Frame ordering
agreed descriptively: F0/F1/F2 active fractions were `0.00/0.50/0.35`, while
refusal-like fractions were `0.40/0.70/0.55`; D0/D1/D2 active fractions were
`0.95/0.95/0.90`, matching refusal-like fractions exactly in aggregate.

Spearman correlation between C and S10 `primary_score` was `rho=0.419483`.
After permuting whole six-frame C vectors between intents, the two-sided
p-value was `0.149585`. The earlier record-level p-value near `2e-6` treated
correlated frames as independent and is not the supported inferential result.
These data motivate a separately preregistered intervention, but association
cannot establish necessity, sufficiency, mediation, or selectivity.

> Caveats that travel with every number above: post-hoc; one annotation pass (manual for the 90 validation records, a Codex-agent pass for the 90 expansion records); join by intent and frame identifiers, text identity not yet hash-verified (Track 1 fixes this); census and generations came from different runs of the same prompts on different devices.

Track 1 resolves the quoted join caveat: all 180 framed-prompt hashes now
match. The annotation-pass and cross-run/device caveats remain. Full public-safe
statistics and provenance hashes are in
`outputs/phase_b_local/s16_phase_b_behavior_concordance.json`.
