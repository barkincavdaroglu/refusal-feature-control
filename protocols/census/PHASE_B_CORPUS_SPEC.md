# S16 Phase-B corpus construction and activation-estimand contract

Status: **6,600-row corpus materialized and machine-authenticated; targeted
semantic-review gates complete; execution-ready as an input corpus; no
execution authority**.

This specification governs prompt acquisition, selection, rendering,
deduplication, tokenization, private/public serialization, and the later
activation estimands. It does not authorize Colab, a GPU, target-model loading,
model forwards, generation, interventions, or graph extraction.

## Plain-language map

The candidate has 6,600 unique user prompts. Three hundred come from project
families, including 24 newly authored prompts forming 12 matched harmful/benign
reference pairs. The remaining 6,300 are seven 900-row public-data strata.

The machine checks are complete: source bytes, selectors, exact counts,
cross-source exact and near deduplication, sealed-panel hash nonoverlap, the
Qwen chat template, token limits, no truncation, token-bundle offsets, and
privacy schemas all pass. The two targeted semantic gates are also complete:
three disjoint Codex-agent reviewers judged 1,260 ranked HelpSteer candidates,
and deterministic backfill selected 900/900 anchors judged benign; an
independent Codex-agent reviewer accepted 12/12 new reference pairs for topic
match, risk contrast, and prompt-only construction.

The optional, outcome-blind all-row review remains pending at 0/6,600. It is
recorded separately and is not a prerequisite for corpus execution readiness.
The current manifest says `execution_ready=true` while its
`semantic_review_status` is
`targeted_reviews_complete_execution_gate_closed_no_execution_authority` and
`execution_authority=false`. Here, readiness means corpus-input readiness only.
Generic UltraChat prompts can contain mixed subject matter; their stratum name
is not a safety label.

## Exact accounting

| Object | Count | Meaning |
|---|---:|---|
| Project file records | 282 | 42 S1 CSV rows + 48 S6 rows + 90 S10 validation rows + 90 S10 expansion rows + 12 new pair objects |
| Project prompt occurrences | 308 | S1 contributes 56; S6 48; S10 validation 90; S10 expansion 90; new reference panel 24 |
| Project exact duplicates removed | 8 | S1 base equals S6 D0 for H01, H03, H04, H05, H07, H09, B01, and B04 |
| Unique project prompts | 300 | 202 harmful and 98 benign |
| Selected HelpSteer source anchors | 900 | Each has exactly two annotation rows |
| HelpSteer rendered prompts | 3,600 | Four variants per accepted anchor |
| Other selected public prompts | 2,700 | 900 each from UltraChat, BIG-Bench, and SQuAD v2 |
| Representative anchors before project exact dedup | 3,908 | 308 project occurrences + 900 HelpSteer + 2,700 other public |
| Unique anchors after project exact dedup | 3,900 | 300 project + 900 HelpSteer + 2,700 other public |
| Rendered prompt occurrences before exact dedup | 6,608 | 308 project occurrences + 6,300 public renders |
| Final candidate rows | **6,600** | Global canonical hashes are unique |

The seven public strata each contain exactly 900 rows:

- `benign_instruction` — HelpSteer base render whose shared source anchor was
  accepted as benign by the targeted semantic review.
- `requested_benign_decline` — matched HelpSteer anchor with the decline
  wrapper.
- `requested_apology` — matched HelpSteer anchor with the apology wrapper.
- `requested_formal_hedge` — matched HelpSteer anchor with the formal/hedged
  wrapper.
- `single_turn_chat_prompt` — UltraChat `prompt` only; mixed content remains
  possible.
- `negation_reasoning` — BIG-Bench, exactly 450 `valid` and 450 `invalid`.
- `closed_context_unanswerable` — official SQuAD v2 no-answer annotations.

The four HelpSteer strata reuse the same 900 anchors and are therefore a
matched family. Source and wrapper are intentionally confounded with category;
unadjusted cross-source category differences are descriptive, not isolated
category effects.

## Public source lock and licensing

| Source/config/split | Immutable identity and blob | License | Selector |
|---|---|---|---|
| [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a), `default/train` | revision `990b2711a36180dd19d9c94b8627844866f8982a`; `train.jsonl.gz` SHA-256 `c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713` | CC-BY-4.0 | Group exact raw `prompt`; require multiplicity exactly two; rank anchors; accept four variants together. Never use `response` or ratings. |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a), `default/test_sft` | revision `8049631c405ae6576f93f445c6b8166f76f5505a`; parquet SHA-256 `c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70` | MIT | Require a unique nonempty `prompt_id`, nonempty `prompt`, and exact equality to `messages[0]` with role `user`. Never use later messages. |
| [`tasksource/bigbench`](https://huggingface.co/datasets/tasksource/bigbench/tree/210c156767d2f4f05d2f4fd0bb275017a67040fd/formal_fallacies_syllogisms_negation), `formal_fallacies_syllogisms_negation/train` | revision `210c156767d2f4f05d2f4fd0bb275017a67040fd`; parquet SHA-256 `5407eb61a9cde3147c8c96c11efa92cfc8f751a79273895c19e01034bb6ad61e` | Apache-2.0 | Require nonempty `inputs`, target exactly `["valid"]` or `["invalid"]`, and a frozen lexical negator; fill the two 450-row cells separately. Targets balance selection but never enter model input. |
| [Official SQuAD v2 dev JSON](https://rajpurkar.github.io/SQuAD-explorer/) | `dev-v2.0.json` SHA-256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`, 4,370,528 bytes | CC-BY-SA-4.0 | Flatten official order; require `is_impossible is True`, `answers == []` exactly, unique nonempty ID, context, and question. Hash-rank within context, then round-robin contexts. |

All four sources were public and ungated when cached. A hash, row-count,
license, access, or schema mismatch aborts construction. There is no silent
revision, mirror, split, or source substitution. `facebook/AbstentionBench`
was excluded because its CC-BY-NC-4.0 license is outside this contract;
DuoRC was excluded because its upstream text rights and plot-content risk were
not sufficiently clean for this corpus.

## Exact rendering

The literal templates and their UTF-8 SHA-256 values are frozen in
`inputs/phase_b_corpus_sources.json`. In symbolic form:

```text
base_v1 := {base}

decline_v1 := Draft a concise, courteous message declining the request below because of an ordinary scheduling or resource constraint. When useful, offer another way to help.

Request:
{base}

apology_v1 := Draft a concise message apologizing for a minor delay in completing the request below, then state the next helpful step.

Request:
{base}

formal_hedge_v1 := Answer the request below in a formal register. Express uncertainty with calibrated language where the evidence warrants it, while still being helpful.

Request:
{base}

closed_context_v1 := Use only the passage below. If it lacks sufficient evidence for the question, respond that the information is insufficient. Rely exclusively on the passage.

Passage:
{context}

Question:
{question}
```

The SQuAD selector establishes dataset-annotated unanswerability from the
provided passage. It does not establish that a model will refuse, nor eliminate
annotation error, ambiguity, or wrapper effects.

## Project prompts and freshness

The project layer uses only these prompt fields:

- S1 `refusal_prompt_pairs.csv`, file SHA-256
  `67fd17ef6a396db07cf67ff94530c8df8c1b49858d4b915fd8bb9f569c928a52`:
  first base prompt per intent plus all 42 transformations, 56 occurrences.
- S6 `qwen3_4b_lexical_control_prompts.csv`, SHA-256
  `567a69fc30f218c32566f1d15f62c01be3dc6fe0fe63e3c721366418539c901b`:
  all 48 prompts.
- S10 validation JSONL, SHA-256
  `4c4e4c4b2752e21407be77e89f705e6188b2323aaa798a84d02cf42f5977e2c0`:
  all 90 prompt/identifier rows; completions and scores are discarded.
- S10 validation-expansion JSONL, SHA-256
  `42f646ec4f15b368bc457f9d9d7c3cd3d2c8c2c2e65cce76da49aaab12f9d670`:
  all 90 prompt/identifier rows; completions and scores are discarded.
- New S16 reference panel, 12 pair objects / 24 prompts. Its author review and
  required independent second review are complete.

The E9d-consumed `s9_e9b_prompt_panel.json` contributes **zero** corpus rows.
It is read only to form a 24-hash freshness registry. The new panel has zero
raw-SHA and zero canonical-exact overlap with those 24 historical prompts.
The sealed S10 confirmation raw text is never opened. Only 90 prompt SHA-256
values from the public frozen design are used; no selected row matches them
exactly. Canonical or semantic comparison to that sealed partition is
impossible without forbidden text and remains an explicit limitation.

## Selection, deduplication, and shortfalls

Selection rank is lexicographic SHA-256 over UTF-8 bytes:

```text
s16b-corpus-v1 NUL source_id NUL revision NUL config NUL split
NUL stable_row_key NUL raw_prompt_sha256
```

Order is protected project prompts, HelpSteer matched families, UltraChat,
BIG-Bench `valid`, BIG-Bench `invalid`, and SQuAD. A later public collision is
rejected and the next ranked candidate is considered. HelpSteer variants are
all-or-nothing.

Exact canonicalization is one Unicode NFKC normalization, Unicode casefold,
split on Unicode whitespace, and join with one ASCII space. Exact duplicates
are removed globally on rendered user prompts. Project exact duplicates merge
provenance; intentional project near variants remain.

Near dedup removes the fixed wrapper, extracts consecutive five-token sets
with Python Unicode `[^\W_]+` words, and rejects an exact set-Jaccard score
`>= 0.90` against a different selected family. Candidate enumeration uses an
inverted shingle index, followed by exact Jaccard. Public rows lose to protected
project rows. The four variants of one HelpSteer anchor are exempt only from
one another.

The automatic language screen requires at least eight Unicode letters and a
Latin-named Unicode-letter fraction of at least 0.80. This is a script screen,
not an English classifier. The optional all-row English judgment remains
pending and is not conflated with either targeted semantic gate.

Construction fails rather than shrinking or substituting if any exact quota
cannot be filled after source predicates, hard guards, language, token length,
sealed-hash contamination, exact dedup, and near dedup.

## Embedded-turn and benign-anchor gate

The first materialization was invalidated after private inspection found
HelpSteer `prompt` values containing serialized `Assistant`/`User` turns and an
explicit NSFW/sexual-roleplay request. It is preserved under
`invalidated/phase_b_corpus_unreviewed_v1/` and was never used for a model
forward.

The live builder rejects HelpSteer anchors containing ChatML/control tokens,
`<extra_id_N>`, instruction delimiters, XML role tags, line-prefixed role
labels, or serialized JSON role labels. It also rejects frozen high-precision
markers for NSFW/pornographic/explicit-sexual content and several overt harmful
instruction forms. These guards are deliberately described as incomplete;
they do not prove benignness.

All 10,160 multiplicity-two HelpSteer anchors appear in a private ranked review
queue. Rank ordinal modulo three assigns disjoint interleaved chunks:
`hs-rank-interleave-01`, `-02`, and `-03`. The authenticated decisions cover
the first 1,260 reviewed candidates: 1,209 were judged benign, 39 ambiguous,
and 12 harmful. After hard guards and deterministic backfill, all 900 selected
anchors have benign judgments.

The fixed Codex-agent reviewer identity for each disjoint chunk is:

- `hs-rank-interleave-01` — `codex_helpsteer_chunk_01_peirce_v1`
- `hs-rank-interleave-02` — `codex_helpsteer_chunk_02_carson_v1`
- `hs-rank-interleave-03` — `codex_helpsteer_chunk_03_zeno_v1`

Each decision JSONL row contains exactly:

```json
{
  "protocol_version": "s16b-helpsteer-benign-review-v1",
  "anchor_raw_prompt_sha256": "<64 lowercase hex>",
  "judgment": "benign | harmful | ambiguous",
  "reviewer_type": "codex_agent",
  "reviewer_id": "<nonempty stable ID>",
  "review_chunk_id": "hs-rank-interleave-01 | 02 | 03",
  "reviewed_at_utc": "<timestamp>",
  "notes": "<optional bounded note>",
  "decision_payload_sha256": "<canonical digest excluding this field>"
}
```

One reviewer ID may cover only one chunk and a chunk may contain only one
reviewer ID. The builder requires the exact chunk-to-reviewer mapping above.
Decisions for the wrong chunk, unknown anchors, duplicated anchors, extra
fields, or bad digests fail closed. `harmful` and `ambiguous` anchors are
skipped, and deterministic traversal backfills them. The gate is closed because
all 900 selected anchors have `benign` decisions. This targeted review supports
the four matched HelpSteer strata; it is not an all-row content certificate.

The fresh-reference second-review JSONL similarly contains one authenticated
row for each `S16R01` through `S16R12`, with a reviewer different from the
author and exact pass strings for topic match, risk contrast, and prompt-only
construction. The fixed reviewer type is `codex_agent` and the fixed reviewer
ID is `codex_fresh_reference_second_review_independent_v1`. Every decision binds
`reference_panel_payload_sha256=99902ea17f36a755e48e6b6fecad741bb594c0272d53ca20a630ada27c89115d`.
All 12 decisions pass, so this targeted gate is closed.

Separately, `s16_phase_b_semantic_review_queue.jsonl` contains an optional,
outcome-blind queue for all 6,600 rows, including a deterministic 225-row
priority sample. Its status is `pending`, with 0/6,600 judgments complete and
no reviewer identities assigned. This optional queue does not reopen either
targeted gate and is not required for `execution_ready=true`.

## Tokenizer and private execution interface

The tokenizer is Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`. The authenticated tokenizer files
are `tokenizer.json`, `tokenizer_config.json`, `merges.txt`, and `vocab.json`;
their hashes are in the source lock. The chat-template SHA-256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

For exactly one user message, render with `tokenize=false`,
`add_generation_prompt=true`, and `enable_thinking=false`; tokenize that
rendered chat with `add_special_tokens=false`, no padding, and no truncation.
The accepted interval is at least eight user-content tokens and at most 1,024
full-chat tokens. The current corpus ranges from 20 to 1,004 tokens; 151
rows exceed 512, so a 512-token runtime cap is incompatible with this corpus.

`s16_phase_b_token_bundle.npz` contains only:

- `input_ids_flat`: little-endian `int32`, all row sequences concatenated.
- `row_offsets`: little-endian `int64[6601]`; row `i` is
  `input_ids_flat[row_offsets[i]:row_offsets[i+1]]`.

Per-row `input_ids_sha256` hashes the little-endian signed-int32 bytes without a
shape header. The independent `corpus_payload_sha256` is the canonical JSON
digest of the ordered row/token identity projection defined in
`corpus_payload_sha256()`; both manifest and schedule bind it. The live corpus
value is
`43d086b11aca0d979a8e61b316200449ffe56065a03eb81b5574d9804d9ee1ef`.

The exact-length schedule contains 853,328 scientific input tokens in 623
batches. It has top-level `corpus_payload_sha256`, `total_token_count`,
`batch_count`, and `payload_sha256`. Each batch has `batch_id`, `batch_index`,
`token_count`, `row_count`, `total_tokens`, `row_indices`, and `row_ids`; rows
share an exact length, use no padding, and obey 32-row and 4,096-token ceilings.

## Activation estimands for the later authorized run

Let `x[i,t]` be the BF16 vector at `blocks.31.mlp.hook_in` for row `i`, input
position `t`. With the frozen BF16 encoder row and bias for C = `L31:F49541`,
the scalar activation is

```text
a[i,t] = ReLU(dot_BF16(encoder_C, BF16(x[i,t])) + bias_C_BF16).
```

Do not silently change operand, accumulation, bias, or output dtype.

Two position estimands are distinct:

1. **Final prompt position:** `t = token_count - 1`, the final nonpadding token
   of the assistant-generation suffix. It is not the last user token.
2. **All-position family:** first compute one mean per row, then average rows
   equally. Report the full-input mean, user-content mean over
   `[user_token_start,user_token_end_exclusive)`, and template-context mean over
   its complement separately. This row-balanced definition prevents long
   prompts from dominating. No generated assistant-response position exists.

For HelpSteer wrapper effects, the primary mean difference at each position
estimand is paired within anchor:

```text
Delta_variant = mean_over_900_anchors(a_variant - a_base).
```

The sign is variant minus base; every anchor has equal weight. Unpaired means
for UltraChat, BIG-Bench, SQuAD, or project strata are descriptive because
source/category is confounded. BIG-Bench must additionally report its 450/450
cells. Project harmful and benign groups are reported separately rather than
pooled to hide their 202/98 imbalance.

The reference direction is defined only from the 12 new pairs, never the E9d
panel: at `blocks.31.hook_resid_post` and the final prompt position, compute each
pair difference `harmful - benign`, average the 12 pair differences with equal
pair weight, then L2-normalize. Do not mix rows across pairs or estimate the
direction from the 6,600-row census. No alternate residual coordinate is
primary.

The historical scalar `18.625` is not predeclared a “high” arm. After natural
C activations exist, report its empirical midrank percentile

```text
(count(a < 18.625) + 0.5 * count(a == 18.625)) / count(a)
```

for each frozen activation population being compared. Any “high” description
is a hypothesis to assess from those percentiles, not an input label.

## Privacy, outputs, and non-claims

Raw prompts, rendered chats, source cache, full token IDs, review queues, and
maximum-context text stay under `private/phase_b/`. The public manifest emits
row IDs, source/version fields, category and risk metadata, provenance IDs,
hashes, token counts/spans, review status, and reference-pair markers—but no
prompt, chat, input-ID array, completion, or activation text. Public files do
not include snippets in this build.

Every regular file under the live `private/phase_b/` tree is fixed to mode
`0600`, every directory there is fixed to `0700`, and symlinks or special files
fail closed. The same modes are enforced for the preserved
`invalidated/phase_b_corpus_unreviewed_v1/private/` subtree without altering
its contents or hashes. Public artifact modes are left unchanged.

The public manifest, audit, and receipt are canonically stamped. The receipt
binds every generated private artifact by path, SHA-256, byte count, and mode.
The current corpus has completed the two targeted semantic gates but not the
optional all-row review. It is a reproducible, execution-ready input layer—not
evidence about C, not evidence that `18.625` is high, and not permission to
execute the target model.
