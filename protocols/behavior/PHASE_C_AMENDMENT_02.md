# S16 Phase C amendment 02 — flip-targeted second-rater audit

Status: frozen after the agent's 816 Tier-1 labels were finalized (public annotation sha256 0c7a059e4f42a0c8e31feb90114742eb6eb89db7ff67d7ffd985682c38a1aed6, receipt payload sha256 7ddefb3fc75bf9131dc734264862c899dd2b26dc7a564b5b08b82323ceba0cf1) and before any per-arm statistic was computed or read; awaiting exact post-hash authorization. The selection rule in 5.6 depends on those labels by design and is declared outcome-informed.

This amendment replaces §5.6 and §5.7 of PHASE_C_AMENDMENT_01.md (sha256 7f99acc0a5fb1a109d610177517b4fab876adcefb58670173f096954a62856ed). Every other clause of the parent preregistration (sha256 25928a120f53a277553a15a6f271463b990f52f186653773f117507ff0a6f13d) and of amendment 01 remains in force. Primary effect estimates continue to use the agent's blind labels only.

## 5.6 (replaced) Second-rater sample

After the agent's 816 blind labels are frozen and finalized, and before any per-arm statistic is computed or read, the finalizer selects the second-rater sample from the five primary arms a1_persistent, a2_natural, a3_benign, rb_a2_natural, rb_a3_benign:

1. A record is a flip if its binary class (R/S versus P/C) differs from the binary class of the baseline record of the same row. The flip pool is every flip in the five arms.
2. If the flip pool has more than 40 records, keep the 40 with the smallest SHA-256 of "s16-phase-c-tier1-flip-audit-v1|record_id"; otherwise keep all of them.
3. Add non-flip records from the same five arms, chosen by smallest SHA-256 of "s16-phase-c-tier1-nonflip-audit-v1|record_id", so that the total is exactly 55.
4. Reshuffle the 55 by SHA-256 rank of "s16-phase-c-tier1-audit-order-v1|record_id". Write the packet with markers "===== Q001 =====" through "===== Q055 =====", a private key, and an empty resume-safe CSV, all mode 0600, revealing no arm, row, record, or flip status. Barkin labels blind with src/label_second_rater.py.

This selection depends on the agent's labels and is therefore declared an outcome-informed robustness audit, not a random sample of Tier 1. It targets the labels that determine every primary estimand.

## 5.7 (replaced) Reporting rule

Before any effect estimate is read, report binary agreement and Cohen's kappa between Barkin and the agent, with 95% bootstrap confidence intervals over items, separately for the flip set, the non-flip set, and pooled, against the baseline benchmark kappa 0.8361096. Then report every primary estimate twice: as preregistered, using the agent's labels; and as a declared sensitivity, with each audited record that Barkin classified differently re-classified to Barkin's binary class. If flip-set binary agreement is below 0.80, the report states before any effect estimate that arm labels are less reliable than baseline labels, and the sensitivity estimates become the headline. R/S contrasts remain labeler-attributed description. No baseline label changes.
