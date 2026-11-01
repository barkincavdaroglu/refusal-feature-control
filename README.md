# Does a “refusal feature” actually control refusal?

Code, frozen numerical observations, annotations, and protocols for causal tests of a candidate feature in Qwen3-4B.

[Read the study](https://www.barkincavdar.com/work/refusal-feature-control/).

## Reproduce the saved-result analyses

Use Python 3.11.7 and the pinned dependencies. The following commands do not load or run a language model:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python run.py verify
python run.py cpu --work /tmp/refusal-feature-analysis
python run.py figures --work /tmp/refusal-feature-analysis
```

Choose an empty scratch directory outside this repository. The runner reconstructs the original relative layout there, checks observation hashes, and preserves the packaged inputs.

The CPU command recomputes the route-development analysis, primary behavioral estimates from 1,020 generation records, and the independent behavioral cross-check. It compares their scientific fields with the frozen results and writes `results/cpu_reproduction.json` in the scratch directory. The figure command rebuilds figures 3, 4, 6, and 7 in its `submission/figures/` directory.

The optional gate-severity analysis uses the same saved inputs:

```sh
python run.py severity --work /tmp/refusal-feature-analysis
```

## Reproduction scope

This package supports saved-observation analysis and figure generation. It is not a complete end-to-end model-run bundle. Prompt/completion text, token and feature input bundles, private blinding lookups, graph tensors, and model weights are not included.

The second-rater audit is a frozen input. Full census/concordance recomputation, independent relabeling, producer authentication, and route replication/fixed-dose reruns require omitted inputs. Their numerical results and scientific protocols are retained, but the commands above do not claim to rerun those experiments. Exact external model and transcoder revisions are listed in `setup/external_assets.json`.

## Files

- `code/`: intervention producers, analyzers, and required import/contract dependencies.
- `data/`: numerical observations, labels, results, and probe parameters. The two TAR files contain 6,600 census shards and 1,020 behavioral records.
- `protocols/`: preregistrations, amendments, failed-label-gate records, and implementation errata.
- `figures/source/`: figure code, inputs, and methods.
- `setup/`: replay helpers, record ordering, environment requirements, and external asset identities.
- `manifest.json`: observation hashes and the path mapping used by the replay runner.
- `SOURCE_MANIFEST.json` and `verify.py`: current package-file integrity and source provenance.
- `licenses/`: attribution for included graph-toolkit code.

Historical hashes embedded in sanitized records identify the original evidence. The root manifest authenticates the distributed derivatives. Integrity checks verify files, not scientific validity. Internal-score changes and changes in generated behavior are separate outcomes; the preregistration amendments and unsuccessful original four-class label gate remain part of the evidence.
