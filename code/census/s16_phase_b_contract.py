#!/usr/bin/env python3
"""Fail-closed schemas and numerical helpers for S16 Phase B.

This module is intentionally CPU-only and side-effect free except for the
explicit atomic JSON helper.  In particular, validating a proposal never
creates execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

EXECUTION_PLAN_SCHEMA = "s16-phase-b-execution-plan/1"
EXECUTION_REQUEST_SCHEMA = "s16-phase-b-execution-request/1"
CORPUS_SCHEMA = 1
BATCH_SCHEDULE_SCHEMA = 1
RUNNING_SCHEMA = "s16-phase-b-running/1"
RESULT_SCHEMA = "s16-phase-b-result/1"
OUTPUT_MANIFEST_SCHEMA = "s16-phase-b-output-manifest/1"
RUNTIME_PROOF_SCHEMA = "s16-phase-b-runtime-proof/1"
PRIVATE_RUNTIME_PROOF_SCHEMA = "s16-phase-b-private-runtime-proof/1"
DEPENDENCY_RUNTIME_PROOF_SCHEMA = "s16-phase-b-dependency-runtime-proof/1"
SESSION_BINDING_SCHEMA = "s16-phase-b-session-binding/1"
TRANSFER_RECEIPT_SCHEMA = "s16-phase-b-transfer-receipt/1"
EXECUTION_RECEIPT_SCHEMA = "s16-phase-b-execution-receipt/1"
DOWNLOAD_RECEIPT_SCHEMA = "s16-phase-b-download-receipt/1"
TEARDOWN_RECEIPT_SCHEMA = "s16-phase-b-teardown-receipt/1"
ABORT_RECEIPT_SCHEMA = "s16-phase-b-abort-receipt/1"
FAILURE_RECEIPT_SCHEMA = "s16-phase-b-operation-failure/1"
ALLOCATION_FAILURE_RECEIPT_SCHEMA = "s16-phase-b-allocation-failure/1"
ALLOCATION_PROVISIONAL_RECEIPT_SCHEMA = "s16-phase-b-allocation-provisional/1"
WATCHDOG_ARM_RECEIPT_SCHEMA = "s16-phase-b-watchdog-arm/1"
ALLOCATION_READY_GATE_SCHEMA = "s16-phase-b-allocation-ready-gate/1"
WATCHDOG_STATUS_RECEIPT_SCHEMA = "s16-phase-b-watchdog-status/1"
CLI_STATE_RECEIPT_SCHEMA = "s16-phase-b-local-cli-state/1"
CLI_STATE_LOCK_SCHEMA = "s16-phase-b-cli-state-lock/1"
LOCAL_RECOMPUTE_RECEIPT_SCHEMA = "s16-phase-b-local-recompute-receipt/1"
SEMANTICS_VERSION = "s16-route-function-phase-b-v1"
SESSION_NAME = "s16-route-function-phase-b-census-v1"
COLAB_CLI_VERSION = "0.6.0"
COLAB_CLI_DISTRIBUTION = "google-colab-cli"
COLAB_CLI_INVOCATION_PATH = "~/.local/bin/colab"
COLAB_CLI_RESOLVED_ENTRYPOINT_PATH = (
    "~/.local/share/uv/tools/google-colab-cli/bin/colab"
)
COLAB_CLI_INTERPRETER_ALIAS_PATH = (
    "~/.local/share/uv/tools/google-colab-cli/bin/python"
)
COLAB_CLI_INTERPRETER_RESOLVED_PATH = (
    "/opt/homebrew/Caskroom/miniconda/base/bin/python3.13"
)
COLAB_CLI_INTERPRETER_IMPLEMENTATION = "CPython"
COLAB_CLI_INTERPRETER_VERSION = "3.13.2"
COLAB_CLI_INTERPRETER_SHA256 = (
    "0f2e48143d6739e280ac0ebab764563a5d17432c5b0d1b5030a6badf2dec7e4d"
)
COLAB_CLI_INTERPRETER_BYTE_COUNT = 5_661_344
WATCHDOG_ABORT_PROBE_TIMEOUT_SECONDS = 120
WATCHDOG_ABORT_RETRY_INTERVAL_SECONDS = 30
WATCHDOG_HEARTBEAT_INTERVAL_SECONDS = 5
WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS = 20
CLI_STATE_LOCK_RELATIVE_PATH = ".s16-phase-b-session-state.lock"
CLI_STATE_LOCK_DISPLAY_PATH = (
    "~/.config/colab-cli/" + CLI_STATE_LOCK_RELATIVE_PATH
)
COLAB_CLI_STATE_ROOT_PATH = "~/.config/colab-cli"
CLI_STATE_LOCK_PROTOCOL = "posix-fcntl-flock-exclusive-v1"
CLI_STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS = 5.0
CLI_STATE_LOCK_POLL_INTERVAL_SECONDS = 0.05
PYTHON_BYTECODE_QUARANTINE_PREFIX = ".s16-phase-b-empty-pycache-"
PYTHON_BYTECODE_QUARANTINE_DISPLAY_TEMPLATE = (
    "~/.config/colab-cli/.s16-phase-b-empty-pycache-<execution-plan-sha256>"
)
IMMUTABLE_JSON_PUBLICATION_PROTOCOL = "direct-openat-o_excl-o_nofollow-fsync-v1"
COLAB_CLI_SITE_PACKAGES_PATH = (
    "~/.local/share/uv/tools/google-colab-cli/"
    "lib/python3.13/site-packages"
)
COLAB_CLI_OAUTH_CONFIG_PATH = (
    COLAB_CLI_SITE_PACKAGES_PATH + "/colab_cli/oauth_config.json"
)
COLAB_CLI_OAUTH_CONFIG_SHA256 = (
    "d3127103bea2884c9f8269d17bcfa74b21356837672bb5b46067cecfe219fc5d"
)
COLAB_CLI_OAUTH_CONFIG_BYTE_COUNT = 438
ISOLATED_PYTHON_FLAGS = ["-I", "-S", "-B"]
PREMODEL_REMOTE_LIFECYCLE_ORDER = [
    "public_stdlib_authority_and_hash_preflight",
    "five_direct_requirement_install",
    "first_exact_session_kernel_restart_after_install",
    "fresh_kernel_top_level_and_observed_distribution_proof",
    "three_private_artifact_upload",
    "second_exact_session_kernel_restart_after_private_upload",
    "private_stdlib_hash_and_mode_preflight",
]
COLAB_CLI_RECORD_SHA256 = (
    "8a7d2e0a8e3af6a28532ecb6aac0132dfdaf9413b817060351a12015287b0e56"
)
COLAB_CLI_ENTRYPOINT_SHA256 = (
    "50ddab851c500bb94ac4bf587470eb455fa9cb3d4ac953ac316d2dd53b9dc61a"
)
COLAB_CLI_ENTRYPOINT_BYTE_COUNT = 352
COLAB_CLI_SOURCE_FILES = {
    "colab_cli/auth.py": "a32d7c91fac00acc90b24b6c70ee80a2c9b8ac0e0a466832196402d97907c579",
    "colab_cli/auto_update.py": "aba759304fe24cb40d1bd18c67ee9c9c1f2414c05c14aa4eec049594c9ad0ac3",
    "colab_cli/cli.py": "de7908d3eb8f21bc083cee20db574d26db0dee215777cc0cf8e4bdfa4fb78915",
    "colab_cli/client.py": "16f31427818e0efea99270acedf8d043a443a4b8488174180f033a108bd7f16c",
    "colab_cli/commands/automation.py": "b622ef1b393b0a664d92493318765c5b50f888ebb1c4f02eb0900a5958004d59",
    "colab_cli/commands/execution.py": "32b1312c3502cd104c96759ea0b21da37083b545b46cfb195bc13e263433dbb4",
    "colab_cli/commands/files.py": "56f82e3b77ed70f765ef81f39efa0a0a6430e76db9ec597ae08aa55ad15def3e",
    "colab_cli/commands/session.py": "59d50c3fe04bfcd1eaa7e2441e79d5e2bed2d0201c469c37e5e614351bbe0c1c",
    "colab_cli/common.py": "277c4d0afe375e482ae1e41d049b73d79af8d7f058cd73e376fa1bddf8f79f0d",
    "colab_cli/contents.py": "eaf42ca4ab3acf0b3cf1b9709897278948df55f60b08e5b996178d406fc6b3b6",
    "colab_cli/history.py": "d1f1ef8a90195dd75dcc1cfb2c1a8a60326afa00b93370f6f1aab041e1ebadb4",
    "colab_cli/runtime.py": "16d9e28cbd9624f05dbe8c58758ace368c0e7492c152b786670b4b5ce91b9e1a",
    "colab_cli/state.py": "5b43ab294b38a7bd450678b9f0762741b0f43b51f036a6c634b385adbb86433e",
}
COLAB_CLI_RECORD_UNHASHED_PATHS = [
    "google_colab_cli-0.6.0.dist-info/RECORD"
]
COLAB_CLI_RECORD_FILE_CLOSURE = {
    "../../../bin/colab": {
        "sha256": "50ddab851c500bb94ac4bf587470eb455fa9cb3d4ac953ac316d2dd53b9dc61a",
        "byte_count": 352,
    },
    "colab_cli/COLAB_SKILL.md": {
        "sha256": "2503b604c2f4313d91322be973d17366acf3f0251211f6fbde6c12561edfc161",
        "byte_count": 9260,
    },
    "colab_cli/README.md": {
        "sha256": "e657a33d47b1104a9f3fe9936cda871deb3d2d2fa85b59c39e0b3ee542394543",
        "byte_count": 8357,
    },
    "colab_cli/auth.py": {
        "sha256": "a32d7c91fac00acc90b24b6c70ee80a2c9b8ac0e0a466832196402d97907c579",
        "byte_count": 9912,
    },
    "colab_cli/auto_update.py": {
        "sha256": "aba759304fe24cb40d1bd18c67ee9c9c1f2414c05c14aa4eec049594c9ad0ac3",
        "byte_count": 8392,
    },
    "colab_cli/cli.py": {
        "sha256": "de7908d3eb8f21bc083cee20db574d26db0dee215777cc0cf8e4bdfa4fb78915",
        "byte_count": 4554,
    },
    "colab_cli/client.py": {
        "sha256": "16f31427818e0efea99270acedf8d043a443a4b8488174180f033a108bd7f16c",
        "byte_count": 10145,
    },
    "colab_cli/commands/__init__.py": {
        "sha256": "036188193fdc7c75748e43ecd28f513f1d291d93f8e7d090a119fca3c2a44741",
        "byte_count": 575,
    },
    "colab_cli/commands/automation.py": {
        "sha256": "b622ef1b393b0a664d92493318765c5b50f888ebb1c4f02eb0900a5958004d59",
        "byte_count": 9395,
    },
    "colab_cli/commands/execution.py": {
        "sha256": "32b1312c3502cd104c96759ea0b21da37083b545b46cfb195bc13e263433dbb4",
        "byte_count": 11524,
    },
    "colab_cli/commands/files.py": {
        "sha256": "56f82e3b77ed70f765ef81f39efa0a0a6430e76db9ec597ae08aa55ad15def3e",
        "byte_count": 6503,
    },
    "colab_cli/commands/run.py": {
        "sha256": "c86f611a7229e222ed16612d273ca578df243826014d4964757c40dd1b1d1285",
        "byte_count": 15800,
    },
    "colab_cli/commands/session.py": {
        "sha256": "59d50c3fe04bfcd1eaa7e2441e79d5e2bed2d0201c469c37e5e614351bbe0c1c",
        "byte_count": 17150,
    },
    "colab_cli/commands/utility.py": {
        "sha256": "2327c63e6734a20dc1cdcc3f168be222c300bedc4b610647373277541865686c",
        "byte_count": 15925,
    },
    "colab_cli/common.py": {
        "sha256": "277c4d0afe375e482ae1e41d049b73d79af8d7f058cd73e376fa1bddf8f79f0d",
        "byte_count": 6368,
    },
    "colab_cli/console.py": {
        "sha256": "d2c9c1e8c42bfa0107ef1fcff3ba4f572581b25484248642b49b88c754bf2c60",
        "byte_count": 5943,
    },
    "colab_cli/contents.py": {
        "sha256": "eaf42ca4ab3acf0b3cf1b9709897278948df55f60b08e5b996178d406fc6b3b6",
        "byte_count": 3026,
    },
    "colab_cli/converter.py": {
        "sha256": "211143cbba34e770ef126c9106d08832b8221ff3a5e768fcfa851ddb0d4adcb2",
        "byte_count": 7118,
    },
    "colab_cli/history.py": {
        "sha256": "d1f1ef8a90195dd75dcc1cfb2c1a8a60326afa00b93370f6f1aab041e1ebadb4",
        "byte_count": 2332,
    },
    "colab_cli/oauth_config.json": {
        "sha256": "d3127103bea2884c9f8269d17bcfa74b21356837672bb5b46067cecfe219fc5d",
        "byte_count": 438,
    },
    "colab_cli/repl.py": {
        "sha256": "6cfd99b677bf101c2fdbf3dffee687d882c53d8f1fee24964880f014bd10f04c",
        "byte_count": 5772,
    },
    "colab_cli/runtime.py": {
        "sha256": "16d9e28cbd9624f05dbe8c58758ace368c0e7492c152b786670b4b5ce91b9e1a",
        "byte_count": 10730,
    },
    "colab_cli/state.py": {
        "sha256": "5b43ab294b38a7bd450678b9f0762741b0f43b51f036a6c634b385adbb86433e",
        "byte_count": 5300,
    },
    "colab_cli/utils.py": {
        "sha256": "09e164e731e19353b735e40074e072a4f5711d8736d50c2087066857b450c857",
        "byte_count": 3590,
    },
    "google_colab_cli-0.6.0.dist-info/INSTALLER": {
        "sha256": "e6184ce10e266134fdcfa401e8f1a95005bcd4f18d16b62b757323e2833fe9a9",
        "byte_count": 2,
    },
    "google_colab_cli-0.6.0.dist-info/METADATA": {
        "sha256": "ba1a6b27f318a5089d559aa27220c0b0e29999f7f989e4b55a7a4f57ee051d5d",
        "byte_count": 9702,
    },
    "google_colab_cli-0.6.0.dist-info/REQUESTED": {
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "byte_count": 0,
    },
    "google_colab_cli-0.6.0.dist-info/WHEEL": {
        "sha256": "99f7cfcbcc019d9427d959c9514e6313df4ab316927e2c8c1d5f58b7468b571b",
        "byte_count": 87,
    },
    "google_colab_cli-0.6.0.dist-info/entry_points.txt": {
        "sha256": "cdea63ff95d9125162fec4ac21febff62e490db3c58fbd67170bc61495389b17",
        "byte_count": 45,
    },
    "google_colab_cli-0.6.0.dist-info/licenses/LICENSE": {
        "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "byte_count": 11358,
    },
}
COLAB_CLI_DIRECT_REQUIREMENT_STRINGS = [
    "click>=8.0",
    "filelock>=3.29.2",
    "google-auth-oauthlib>=1.3.0",
    "google-auth>=2.49.1",
    "html2text>=2024.2.26",
    "jupyter-kernel-client",
    "nbformat>=5.10.4",
    "packaging>=24.0",
    "prompt-toolkit>=3.0.52",
    "pydantic>=2.12.5",
    "pygments>=2.19.2",
    "requests>=2.32.5",
    "rich>=14.3.3",
    "typer>=0.24.1",
    "typing-extensions>=4.0",
    "websocket-client>=1.0",
]
COLAB_CLI_DIRECT_RUNTIME_DEPENDENCIES = {
    "click": {
        "distribution_name": "click",
        "version": "8.4.2",
        "record_sha256": "f14e3990f101ef4a1dd09903eb3f859f233ec1d9bd9a00e4470f207568804d12",
    },
    "filelock": {
        "distribution_name": "filelock",
        "version": "3.32.2",
        "record_sha256": "29872bd99dcbb773451d0a701b019ff5df251ebacf02be4d103b001619b8a4b1",
    },
    "google-auth": {
        "distribution_name": "google-auth",
        "version": "2.56.3",
        "record_sha256": "7d1d188ef9a61d5fc308a2b72ec8c858c3c59c50dda1bb157902ea8e8b15551a",
    },
    "google-auth-oauthlib": {
        "distribution_name": "google-auth-oauthlib",
        "version": "1.4.0",
        "record_sha256": "f0d795f568293ea10ff54d81f14d3dc788c085cbcb7117e08d058bbd8aaf89c1",
    },
    "html2text": {
        "distribution_name": "html2text",
        "version": "2025.4.15",
        "record_sha256": "fa4cd8916e375028c82d183f4376d92c1fbe6e8b5c0204f33dae93cb3fa6e44c",
    },
    "jupyter-kernel-client": {
        "distribution_name": "jupyter_kernel_client",
        "version": "0.8.0",
        "record_sha256": "2b2c70afcd20094f2eb4984e27615729366c0241c21df5d4d7b7cb0b07da51b0",
    },
    "nbformat": {
        "distribution_name": "nbformat",
        "version": "5.11.0",
        "record_sha256": "172c4e4059f4cabdf243775052dbbb7d19027d8f2732f69c863fae4505e3bc72",
    },
    "packaging": {
        "distribution_name": "packaging",
        "version": "26.3",
        "record_sha256": "1b8655965b0d83e9a067fd7f84b977bd1e117e0fb9f2874b2572781bdc9e2147",
    },
    "prompt-toolkit": {
        "distribution_name": "prompt_toolkit",
        "version": "3.0.53",
        "record_sha256": "a40bdf71ed62886d4459e1462979f862f83d6fde483551fb8e547c0a6764df1d",
    },
    "pydantic": {
        "distribution_name": "pydantic",
        "version": "2.13.4",
        "record_sha256": "d0cbe0f34b294885175920bc8f2ece0b3ed6d9874347dc9cfb2caf4931329272",
    },
    "pygments": {
        "distribution_name": "Pygments",
        "version": "2.20.0",
        "record_sha256": "a8530fa10da3a55ff7cf67b4881abc3a11d20d9ba2b3fbf3d051a298cda8b9f0",
    },
    "requests": {
        "distribution_name": "requests",
        "version": "2.34.2",
        "record_sha256": "fc8c5216d0e865d815bd04cc240b5adc5d56f6c3dbe9aef3b4ea3b83d735bdfe",
    },
    "rich": {
        "distribution_name": "rich",
        "version": "15.0.0",
        "record_sha256": "1b725253b1bd316b53169a3a0b35d03d6c81bad9009ddae73d8645fb3cd3f292",
    },
    "typer": {
        "distribution_name": "typer",
        "version": "0.27.1",
        "record_sha256": "be236b57d5e6251a1ce78f56c256987a9eb1b68abac69ad71b082dce159179be",
    },
    "typing-extensions": {
        "distribution_name": "typing_extensions",
        "version": "4.16.0",
        "record_sha256": "9e07fea845cac76293d927afa7157a14ecba39b318bc8af236b8de530b7656c1",
    },
    "websocket-client": {
        "distribution_name": "websocket-client",
        "version": "1.9.0",
        "record_sha256": "698074ef3f56eb7c9939f1456e2892b9530541ff7a9d7f8abe471d118a22ba5c",
    },
}
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_FILE_CLOSURE_SHA256 = (
    "8dbf82d4fe6c26d3cf19b5cee9b74300b7981908c137b164a784c69d04b28321"
)
MODEL_FILES = {
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
    "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MODEL_FILE_SIZES = {
    "config.json": 726,
    "generation_config.json": 239,
    "merges.txt": 1_671_853,
    "model-00001-of-00003.safetensors": 3_957_900_840,
    "model-00002-of-00003.safetensors": 3_987_450_520,
    "model-00003-of-00003.safetensors": 99_630_640,
    "model.safetensors.index.json": 32_819,
    "tokenizer.json": 11_422_654,
    "tokenizer_config.json": 9_732,
    "vocab.json": 2_776_833,
}
TRANSCODER_REVISION = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
FEATURE_BUNDLE_SHA256 = (
    "8a2de5629bcdbd138d23b9ed9705b6c4a2ac78d9995912ded0c820a0b0f23ffc"
)
FEATURE_MANIFEST_SHA256 = (
    "c19d0722826a475fced422f73a87f9173948c43cc26d42a3751c34ecf337ef20"
)
FEATURE_NAME = "L31:F49541"
FEATURE_INDEX = 49541
FEATURE_HOOK = "blocks.31.mlp.hook_in"
PRIMARY_REFERENCE_HOOK = "blocks.31.hook_resid_post"
SECONDARY_REFERENCE_HOOKS = (
    "blocks.31.hook_resid_pre",
    FEATURE_HOOK,
)
EXPECTED_ROW_COUNT = 6600
EXPECTED_REFERENCE_PAIR_COUNT = 12
MAX_INPUT_TOKENS = 1024
MAX_BATCH_ROWS = 32
MAX_BATCH_TOKENS = 4096
MAX_OUTPUT_BYTES = 1 << 30
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2522210527409210963
BOOTSTRAP_GENERATOR = "numpy.random.PCG64"
HELPSTEER_WRAPPER_VARIANTS = ("decline_v1", "apology_v1", "formal_hedge_v1")
HELPSTEER_ANCHOR_COUNT = 900
HELPSTEER_FAMILYWISE_ALPHA = 0.05
HELPSTEER_FAMILYWISE_METHOD = (
    "bonferroni_adjusted_paired_anchor_percentile_bootstrap"
)
HISTORICAL_DOSE = 18.625
PACKAGE_VERSIONS = {
    "numpy": "2.2.6",
    "torch": "2.11.0+cu128",
    "transformers": "4.57.3",
    "transformer-lens": "3.2.1",
    "safetensors": "0.8.0",
    "huggingface-hub": "0.36.2",
}
REMOTE_DIRECT_INSTALL_REQUIREMENT_LINES = [
    "numpy==2.2.6",
    "transformers==4.57.3",
    "transformer-lens==3.2.1",
    "safetensors==0.8.0",
    "huggingface-hub==0.36.2",
]
AUTHORITY_FLAG_KEYS = frozenset(
    {
        "researcher_approved",
        "receipt_grants_execution_authority",
        "private_token_upload_authorized",
        "private_research_artifact_upload_authorized",
        "a100_allocation_authorized",
        "target_model_load_authorized",
        "target_model_forward_authorized",
        "raw_prompt_upload_authorized",
        "intervention_authorized",
        "generation_authorized",
        "backward_authorized",
        "sealed_s10_confirmation_authorized",
        "phase_c_authorized",
        "phase_d_authorized",
    }
)
PROPOSAL_REQUESTED_SCOPE = {
    "one_fresh_a100_session": True,
    "private_pretokenized_upload": True,
    "private_research_feature_bundle_upload": True,
    "forward_only_activation_census": True,
    "local_recomputation_then_exact_session_teardown": True,
}
NO_AUTHORITY_PROPOSAL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "execution_plan_payload_sha256",
        "challenge",
        "required_confirmation_text",
        "required_confirmation_text_sha256",
        "requested_scope",
        "payload_sha256",
        *AUTHORITY_FLAG_KEYS,
    }
)
AUTHORIZED_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "execution_plan_payload_sha256",
        "proposal_payload_sha256",
        "challenge",
        "approval_reference",
        "confirmation_text_sha256",
        "payload_sha256",
        *AUTHORITY_FLAG_KEYS,
    }
)
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
SCIENTIFIC_IMPORT_PROOF = [
    "numpy",
    "torch",
    "transformers",
    "transformer_lens",
    "safetensors",
    "huggingface_hub",
]
PUBLIC_RUNTIME_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at_utc",
        "execution_plan_payload_sha256",
        "execution_request_payload_sha256",
        "proposal_payload_sha256",
        "allocation_provisional_receipt_payload_sha256",
        "allocation_ready_gate_payload_sha256",
        "watchdog_arm_receipt_payload_sha256",
        "session_binding_payload_sha256",
        "uploaded_public_input_hashes",
        "private_input_roles_confirmed_absent",
        "scientific_modules_imported",
        "model_loaded",
        "model_forwards",
        "payload_sha256",
    }
)
DEPENDENCY_RUNTIME_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at_utc",
        "python",
        "python_implementation",
        "platform",
        "device",
        "cuda_total_memory_gib",
        "torch_cuda",
        "device_capability",
        "cudnn_version",
        "cublas_workspace_config",
        "cudnn_benchmark",
        "allow_tf32",
        "deterministic_algorithms",
        "direct_install_requirement_lines",
        "top_level_package_versions",
        "installed_distribution_inventory",
        "installed_distribution_count",
        "transitive_versions_status",
        "wheel_hashes_status",
        "scientific_imports_completed",
        "execution_plan_payload_sha256",
        "execution_request_payload_sha256",
        "proposal_payload_sha256",
        "allocation_provisional_receipt_payload_sha256",
        "allocation_ready_gate_payload_sha256",
        "watchdog_arm_receipt_payload_sha256",
        "session_binding_payload_sha256",
        "public_preflight_payload_sha256",
        "fresh_kernel_after_dependency_install",
        "model_loaded",
        "model_forwards",
        "payload_sha256",
    }
)
PRIVATE_RUNTIME_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_at_utc",
        "execution_plan_payload_sha256",
        "execution_request_payload_sha256",
        "proposal_payload_sha256",
        "allocation_provisional_receipt_payload_sha256",
        "allocation_ready_gate_payload_sha256",
        "watchdog_arm_receipt_payload_sha256",
        "session_binding_payload_sha256",
        "uploaded_private_input_hashes",
        "private_input_roles",
        "public_preflight_payload_sha256",
        "dependency_proof_payload_sha256",
        "scientific_modules_imported_by_this_process",
        "model_loaded",
        "model_forwards",
        "payload_sha256",
    }
)
PREMODEL_PROOF_BINDING_KEYS = frozenset(
    {
        "preinstall_payload_sha256",
        "dependency_payload_sha256",
        "private_preflight_payload_sha256",
    }
)
REQUIRED_INPUT_ROLES = {
    "analyzer",
    "assignment_probe",
    "authority_materializer",
    "batch_schedule",
    "cli_bootstrap",
    "colab_controller",
    "corpus_manifest",
    "corpus_audit",
    "corpus_receipt",
    "corpus_specification",
    "execution_plan_builder",
    "feature_bundle",
    "feature_manifest",
    "finalize_launcher",
    "forward_runner",
    "forward_specification",
    "local_verifier",
    "packager",
    "phase_b_contract",
    "preregistration",
    "probe_direction",
    "remote_bootstrap",
    "remote_dependency_proof",
    "remote_stdlib_preflight",
    "run_launcher",
    "runtime_requirements",
    "token_bundle",
}
MODEL_SNAPSHOT_PROOF_SCHEMA = "s16-phase-b-model-snapshot-proof/1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


class ContractError(ValueError):
    """One S16 Phase-B contract or authenticated artifact differs."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scientific_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"payload_sha256", "scientific_payload_sha256", "created_at_utc"}
    }
    return canonical_sha256(body)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_distribution_name(name: str) -> str:
    """Return the PyPA comparison form without importing packaging."""

    return re.sub(r"[-_.]+", "-", name).lower()


def ensure_private_directory(path: pathlib.Path) -> None:
    """Create/check a runtime-owned private directory with exact mode 0700."""

    if path.is_symlink():
        raise ContractError(f"private directory is unsafe: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    os.chmod(path, PRIVATE_DIRECTORY_MODE)
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != PRIVATE_DIRECTORY_MODE:
        raise ContractError(f"private directory mode differs: {path}")


def require_private_file(path: pathlib.Path) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != PRIVATE_FILE_MODE
    ):
        raise ContractError(f"private file mode differs: {path}")


def require_private_directory(path: pathlib.Path) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) != PRIVATE_DIRECTORY_MODE
    ):
        raise ContractError(f"private directory mode differs: {path}")


def require_no_symlink_components(root: pathlib.Path, path: pathlib.Path) -> None:
    """Require every component below an authenticated root to be a real path."""

    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ContractError(f"path is outside authenticated root: {path}") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"path contains an unsafe symlink component: {path}")


def require_private_ancestry(root: pathlib.Path, path: pathlib.Path) -> None:
    """Require the file and each directory from its `private` boundary downward."""

    require_no_symlink_components(root, path)
    relative = path.absolute().relative_to(root.absolute())
    try:
        private_index = relative.parts.index("private")
    except ValueError as error:
        raise ContractError(f"private input lacks a private path boundary: {path}") from error
    current = root.joinpath(*relative.parts[:private_index])
    for part in relative.parts[private_index:-1]:
        current = current / part
        require_private_directory(current)
    require_private_file(path)


def atomic_json(
    path: pathlib.Path,
    payload: Mapping[str, Any],
    *,
    private_parent: bool = False,
) -> None:
    data = dict(payload)
    data["payload_sha256"] = canonical_sha256(data)
    if path.is_symlink():
        raise ContractError(f"authenticated JSON destination is unsafe: {path}")
    if private_parent:
        ensure_private_directory(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, PRIVATE_FILE_MODE)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def atomic_json_new(
    path: pathlib.Path,
    payload: Mapping[str, Any],
    *,
    private_parent: bool = False,
) -> None:
    """Durably publish authenticated JSON exactly once without replacement."""

    data = dict(payload)
    data["payload_sha256"] = canonical_sha256(data)
    absolute_path = path.absolute()
    require_no_symlink_components(pathlib.Path(absolute_path.anchor), absolute_path.parent)
    if path.is_symlink() or path.exists():
        raise ContractError(f"immutable authenticated JSON already exists: {path}")
    if private_parent:
        ensure_private_directory(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ContractError("immutable JSON publication requires no-follow directory open")
    encoded = (
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    parent_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise ContractError(
                f"immutable authenticated JSON already exists: {path}"
            ) from error
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
        linked = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_uid != os.getuid()
            or linked.st_nlink != 1
            or stat.S_IMODE(linked.st_mode) != PRIVATE_FILE_MODE
        ):
            raise ContractError(f"immutable authenticated JSON inode differs: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def load_authenticated_json(
    path: pathlib.Path, *, schema: Any | None = None
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"authenticated JSON path is absent or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read authenticated JSON {path}: {error}") from error
    if not isinstance(payload, dict) or canonical_sha256(payload) != payload.get(
        "payload_sha256"
    ):
        raise ContractError(f"JSON authentication failed: {path}")
    if schema is not None and payload.get("schema_version") != schema:
        raise ContractError(f"JSON schema differs: {path}")
    return payload


def load_immutable_authenticated_json(
    path: pathlib.Path, *, schema: Any | None = None
) -> dict[str, Any]:
    """Load a write-once receipt through an authenticated private inode."""

    absolute_path = path.absolute()
    require_no_symlink_components(pathlib.Path(absolute_path.anchor), absolute_path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ContractError("immutable JSON loading requires O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ContractError(
            f"immutable authenticated JSON path is absent or unsafe: {path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        linked = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != PRIVATE_FILE_MODE
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise ContractError(f"immutable authenticated JSON inode differs: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read immutable authenticated JSON {path}") from error
    if not isinstance(payload, dict) or canonical_sha256(payload) != payload.get(
        "payload_sha256"
    ):
        raise ContractError(f"JSON authentication failed: {path}")
    if schema is not None and payload.get("schema_version") != schema:
        raise ContractError(f"JSON schema differs: {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _exact_bool(value: Any, expected: bool) -> bool:
    return value is expected


def _require_exact_keys(
    value: Any, expected: set[str] | frozenset[str], label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ContractError(f"Phase-B {label} key closure differs")


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError("UTC timestamp differs")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ContractError("UTC timestamp differs") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError("UTC timestamp differs")
    return parsed


def validate_allocation_provisional_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "allocated_at_utc",
            "lease_expires_at_utc",
            "absolute_allocation_lease_seconds",
            "session_name",
            "endpoint_sha256",
            "accelerator",
            "variant",
            "preallocation_assignment_count",
            "postallocation_assignment_count",
            "execution_plan_payload_sha256",
            "execution_request_payload_sha256",
            "scientific_use_authorized",
            "watchdog_arm_required",
            "global_sync_or_prune_used",
            "payload_sha256",
        },
        "allocation provisional receipt",
    )
    start = parse_utc(receipt.get("allocated_at_utc"))
    deadline = parse_utc(receipt.get("lease_expires_at_utc"))
    parse_utc(receipt.get("created_at_utc"))
    seconds = plan.get("resource_contract", {}).get(
        "absolute_allocation_lease_seconds"
    )
    if (
        receipt.get("schema_version") != ALLOCATION_PROVISIONAL_RECEIPT_SCHEMA
        or canonical_sha256(receipt) != receipt.get("payload_sha256")
        or receipt.get("status")
        != "one_new_s16_a100_provisionally_allocated_watchdog_not_yet_accepted"
        or receipt.get("session_name") != SESSION_NAME
        or receipt.get("accelerator") != "A100"
        or receipt.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or receipt.get("execution_request_payload_sha256")
        != request.get("payload_sha256")
        or receipt.get("absolute_allocation_lease_seconds") != seconds
        or not _is_sha256(receipt.get("endpoint_sha256"))
        or not isinstance(receipt.get("variant"), str)
        or isinstance(receipt.get("preallocation_assignment_count"), bool)
        or not isinstance(receipt.get("preallocation_assignment_count"), int)
        or receipt.get("preallocation_assignment_count", -1) < 0
        or receipt.get("postallocation_assignment_count")
        != receipt.get("preallocation_assignment_count", -1) + 1
        or receipt.get("scientific_use_authorized") is not False
        or receipt.get("watchdog_arm_required") is not True
        or receipt.get("global_sync_or_prune_used") is not False
        or (deadline - start).total_seconds() != seconds
    ):
        raise ContractError("Phase-B provisional allocation receipt differs")
    return start, deadline


def validate_watchdog_arm_receipt(
    arm: Mapping[str, Any],
    *,
    provisional: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    validate_allocation_provisional_receipt(provisional, plan=plan, request=request)
    _require_exact_keys(
        arm,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "watchdog_pid",
            "lease_expires_at_utc",
            "endpoint_sha256",
            "execution_plan_payload_sha256",
            "execution_request_payload_sha256",
            "allocation_provisional_receipt_payload_sha256",
            "watchdog_status_receipt_name",
            "scientific_completion_required",
            "abort_retry_until_absence_proven",
            "abort_probe_timeout_seconds",
            "abort_retry_interval_seconds",
            "heartbeat_interval_seconds",
            "heartbeat_max_age_seconds",
            "payload_sha256",
        },
        "immutable watchdog arm receipt",
    )
    parse_utc(arm.get("created_at_utc"))
    if (
        arm.get("schema_version") != WATCHDOG_ARM_RECEIPT_SCHEMA
        or canonical_sha256(arm) != arm.get("payload_sha256")
        or arm.get("status") != "absolute_allocation_watchdog_armed_immutable"
        or isinstance(arm.get("watchdog_pid"), bool)
        or not isinstance(arm.get("watchdog_pid"), int)
        or arm.get("watchdog_pid", 0) <= 0
        or arm.get("lease_expires_at_utc") != provisional.get("lease_expires_at_utc")
        or arm.get("endpoint_sha256") != provisional.get("endpoint_sha256")
        or arm.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or arm.get("execution_request_payload_sha256") != request.get("payload_sha256")
        or arm.get("allocation_provisional_receipt_payload_sha256")
        != provisional.get("payload_sha256")
        or not isinstance(arm.get("watchdog_status_receipt_name"), str)
        or pathlib.PurePath(arm.get("watchdog_status_receipt_name", "")).name
        != arm.get("watchdog_status_receipt_name")
        or arm.get("scientific_completion_required") is not False
        or arm.get("abort_retry_until_absence_proven") is not True
        or arm.get("abort_probe_timeout_seconds")
        != WATCHDOG_ABORT_PROBE_TIMEOUT_SECONDS
        or arm.get("abort_retry_interval_seconds")
        != WATCHDOG_ABORT_RETRY_INTERVAL_SECONDS
        or arm.get("heartbeat_interval_seconds") != WATCHDOG_HEARTBEAT_INTERVAL_SECONDS
        or arm.get("heartbeat_max_age_seconds") != WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS
    ):
        raise ContractError("Phase-B immutable watchdog arm receipt differs")


def validate_allocation_ready_gate(
    gate: Mapping[str, Any],
    *,
    provisional: Mapping[str, Any],
    arm: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    start, deadline = validate_allocation_provisional_receipt(
        provisional, plan=plan, request=request
    )
    validate_watchdog_arm_receipt(
        arm, provisional=provisional, plan=plan, request=request
    )
    _require_exact_keys(
        gate,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "allocated_at_utc",
            "lease_expires_at_utc",
            "absolute_allocation_lease_seconds",
            "session_name",
            "endpoint_sha256",
            "accelerator",
            "variant",
            "execution_plan_payload_sha256",
            "execution_request_payload_sha256",
            "allocation_provisional_receipt_payload_sha256",
            "watchdog_arm_receipt_payload_sha256",
            "watchdog_pid",
            "watchdog_status_receipt_name",
            "scientific_use_requires_unexpired_lease_and_live_watchdog",
            "payload_sha256",
        },
        "allocation ready gate",
    )
    parse_utc(gate.get("created_at_utc"))
    if (
        gate.get("schema_version") != ALLOCATION_READY_GATE_SCHEMA
        or canonical_sha256(gate) != gate.get("payload_sha256")
        or gate.get("status") != "allocation_ready_after_authenticated_watchdog_arm"
        or gate.get("allocated_at_utc") != provisional.get("allocated_at_utc")
        or gate.get("lease_expires_at_utc") != provisional.get("lease_expires_at_utc")
        or gate.get("absolute_allocation_lease_seconds")
        != provisional.get("absolute_allocation_lease_seconds")
        or gate.get("session_name") != SESSION_NAME
        or gate.get("endpoint_sha256") != provisional.get("endpoint_sha256")
        or gate.get("accelerator") != provisional.get("accelerator")
        or gate.get("variant") != provisional.get("variant")
        or gate.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or gate.get("execution_request_payload_sha256") != request.get("payload_sha256")
        or gate.get("allocation_provisional_receipt_payload_sha256")
        != provisional.get("payload_sha256")
        or gate.get("watchdog_arm_receipt_payload_sha256") != arm.get("payload_sha256")
        or gate.get("watchdog_pid") != arm.get("watchdog_pid")
        or gate.get("watchdog_status_receipt_name")
        != arm.get("watchdog_status_receipt_name")
        or gate.get("scientific_use_requires_unexpired_lease_and_live_watchdog")
        is not True
    ):
        raise ContractError("Phase-B allocation ready gate differs")
    return start, deadline


def allocation_evidence_bindings(gate: Mapping[str, Any]) -> dict[str, str]:
    """Return the three immutable receipt identities carried by descendants."""

    return {
        "allocation_provisional_receipt_payload_sha256": gate.get(
            "allocation_provisional_receipt_payload_sha256"
        ),
        "allocation_ready_gate_payload_sha256": gate.get("payload_sha256"),
        "watchdog_arm_receipt_payload_sha256": gate.get(
            "watchdog_arm_receipt_payload_sha256"
        ),
    }


ALLOCATION_EVIDENCE_BINDING_KEYS = frozenset(
    {
        "allocation_provisional_receipt_payload_sha256",
        "allocation_ready_gate_payload_sha256",
        "watchdog_arm_receipt_payload_sha256",
    }
)


def validate_allocation_evidence_hash_fields(
    record: Mapping[str, Any], *, label: str
) -> None:
    """Require the complete three-hash immutable allocation lineage."""

    if any(
        not _is_sha256(record.get(key))
        for key in ALLOCATION_EVIDENCE_BINDING_KEYS
    ):
        raise ContractError(f"Phase-B {label} allocation-evidence hashes differ")


def validate_allocation_evidence_bindings(
    record: Mapping[str, Any], gate: Mapping[str, Any], *, label: str
) -> None:
    validate_allocation_evidence_hash_fields(record, label=label)
    expected = allocation_evidence_bindings(gate)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ContractError(f"Phase-B {label} allocation evidence differs")


def validate_live_watchdog_status(
    status: Mapping[str, Any],
    *,
    provisional: Mapping[str, Any],
    arm: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    now: datetime | None = None,
) -> None:
    validate_watchdog_arm_receipt(
        arm, provisional=provisional, plan=plan, request=request
    )
    _require_exact_keys(
        status,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "heartbeat_at_utc",
            "watchdog_pid",
            "lease_expires_at_utc",
            "endpoint_sha256",
            "execution_plan_payload_sha256",
            "execution_request_payload_sha256",
            "allocation_provisional_receipt_payload_sha256",
            "watchdog_arm_receipt_payload_sha256",
            "exact_assignment_absence_proven",
            "abort_retry_continues",
            "scientific_completion_required",
            "payload_sha256",
        },
        "live watchdog status",
    )
    created = parse_utc(status.get("created_at_utc"))
    heartbeat = parse_utc(status.get("heartbeat_at_utc"))
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise ContractError("Phase-B watchdog clock is timezone-naive")
    age = (current.astimezone(timezone.utc) - heartbeat).total_seconds()
    if (
        status.get("schema_version") != WATCHDOG_STATUS_RECEIPT_SCHEMA
        or canonical_sha256(status) != status.get("payload_sha256")
        or status.get("status") != "absolute_allocation_watchdog_active"
        or created > heartbeat
        or age < -1.0
        or age > WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS
        or status.get("watchdog_pid") != arm.get("watchdog_pid")
        or status.get("lease_expires_at_utc") != provisional.get("lease_expires_at_utc")
        or status.get("endpoint_sha256") != provisional.get("endpoint_sha256")
        or status.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or status.get("execution_request_payload_sha256") != request.get("payload_sha256")
        or status.get("allocation_provisional_receipt_payload_sha256")
        != provisional.get("payload_sha256")
        or status.get("watchdog_arm_receipt_payload_sha256") != arm.get("payload_sha256")
        or status.get("exact_assignment_absence_proven") is not False
        or status.get("abort_retry_continues") is not True
        or status.get("scientific_completion_required") is not False
    ):
        raise ContractError("Phase-B live watchdog status differs")


def remaining_allocation_lease_seconds(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    now: datetime | None = None,
) -> float:
    del plan, request
    deadline = parse_utc(receipt.get("lease_expires_at_utc"))
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise ContractError("Phase-B lease clock is timezone-naive")
    return (deadline - current.astimezone(timezone.utc)).total_seconds()


def validate_execution_plan(plan: Mapping[str, Any]) -> None:
    _require_exact_keys(
        plan,
        {
            "schema_version",
            "semantics_version",
            "status",
            "model",
            "feature",
            "reference_directions",
            "corpus",
            "batching",
            "resource_contract",
            "runtime",
            "forward_accounting",
            "estimands",
            "prespecified_contrasts",
            "bootstrap",
            "privacy_and_transfer",
            "session_isolation",
            "colab_cli",
            "inputs",
            "payload_sha256",
        },
        "execution-plan top-level",
    )
    nested_keys = {
        "model": {
            "id", "revision", "file_closure_sha256", "files", "dtype",
            "snapshot_directory_name", "exact_file_set_required", "proof_schema",
        },
        "feature": {
            "name", "index", "activation_hook", "bundle_sha256",
            "manifest_sha256", "transcoder_revision", "activation_formula",
            "formatted_final_semantics", "all_token_semantics",
        },
        "reference_directions": {
            "primary_hook", "secondary_hooks", "orientation", "pair_balanced",
            "pair_count", "reference_panel", "probe_coordinate_caveat",
        },
        "corpus": {
            "row_count", "total_token_count", "batch_count",
            "corpus_payload_sha256", "manifest_payload_sha256",
            "batch_schedule_payload_sha256", "semantic_review_status",
            "raw_prompt_text_transferred", "pretokenized", "no_padding",
        },
        "batching": {
            "max_input_tokens", "max_batch_rows", "max_batch_tokens",
            "equal_length_only", "padding_allowed", "truncation_allowed",
            "batch_size_one_fallback_on_failed_parity_audit",
        },
        "resource_contract": {
            "session_name", "requested_accelerator", "maximum_gpu_sessions",
            "maximum_authenticated_model_loads", "minimum_cuda_memory_gib",
            "hard_wall_seconds", "absolute_allocation_lease_seconds",
            "maximum_output_bytes", "generation_allowed", "intervention_allowed",
            "backward_allowed", "sealed_s10_confirmation_allowed",
        },
        "runtime": {
            "seed", "audit_row_indices", "batch_parity_abs_tolerance",
            "direct_install_requirement_lines",
            "post_install_top_level_package_versions",
            "full_installed_distribution_version_inventory_required",
            "transitive_versions_preregistered", "wheel_hashes_preregistered",
            "python_implementation", "torch_cuda_version",
            "model_snapshot_proof_schema", "cublas_workspace_config", "model_eval",
            "gradients_enabled", "allow_tf32", "deterministic_algorithms",
            "kernel_restart_after_install_required",
            "kernel_restart_after_private_upload_before_private_preflight_required",
            "premodel_remote_lifecycle_order",
            "local_files_only_after_authenticated_snapshot",
        },
        "forward_accounting": {
            "unique_scientific_prompt_equivalents",
            "unique_scientific_token_equivalents",
            "planned_scientific_invocations",
            "planned_fallback_scientific_invocations",
            "maximum_scientific_invocations", "audit_invocations",
            "audit_prompt_equivalents", "audit_token_equivalents",
            "attempted_scientific_token_equivalent_ceiling",
            "attempted_total_token_equivalent_ceiling",
            "attempted_total_invocation_ceiling",
            "attempted_prompt_equivalent_ceiling",
        },
        "bootstrap": {
            "replicates", "seed", "generator", "unit", "lopo",
            "helpsteer_anchor_replicates", "helpsteer_anchor_seed",
            "helpsteer_anchor_generator", "helpsteer_familywise_alpha",
            "helpsteer_familywise_method",
        },
        "privacy_and_transfer": {
            "raw_prompt_or_chat_text_uploaded", "private_token_inputs",
            "private_research_artifact_inputs", "raw_corpus_manifest_uploaded",
            "semantic_review_queue_uploaded",
            "colab_source_or_arguments_contain_private_tokens",
            "private_inputs_uploaded_only_after_dependency_proof",
            "second_stdlib_private_preflight_required",
            "private_directory_mode_octal", "private_file_mode_octal",
        },
        "session_isolation": {
            "fresh_named_session_required", "q004_session_reuse_allowed",
            "exact_endpoint_bound_before_private_upload", "global_sync_or_prune_allowed",
            "local_recomputation_before_exact_session_teardown",
            "absolute_allocation_lease_seconds", "local_exact_assignment_watchdog_required",
            "watchdog_abort_probe_timeout_seconds", "watchdog_abort_retry_interval_seconds",
            "watchdog_abort_retry_until_absence_proven",
            "watchdog_ambiguity_receipt_after_each_failed_attempt",
            "watchdog_process_and_host_liveness_trusted_local_tcb",
            "watchdog_survives_host_or_process_failure_claimed",
            "emergency_abort_requires_scientific_completion",
            "unattributed_server_orphan_cannot_be_excluded_by_local_name_probe",
            "fixed_cli_state_lock_path", "cli_state_lock_protocol",
            "cli_state_lock_acquire_timeout_seconds",
            "cli_state_lock_poll_interval_seconds",
            "cli_state_lock_file_mode_octal", "cli_state_lock_parent_mode_octal",
            "session_name_probe_mutation_probe_lock_required",
            "cooperating_s16_process_exclusivity_local_tcb",
            "arbitrary_same_user_raw_cli_excluded_claimed",
            "watchdog_exact_endpoint_abort_bounded_lock_then_bypass",
            "watchdog_lock_bypass_may_race_inflight_cli_mutation",
            "cli_state_lock_marker_authority_specific",
            "lock_inherited_by_direct_child_until_exit",
            "lock_holder_process_liveness_trusted_local_tcb",
            "lock_survives_controller_and_direct_child_failure_claimed",
            "provisional_allocation_is_scientific_authority",
            "immutable_watchdog_arm_receipt_required",
            "allocation_ready_gate_required_for_science",
            "allocation_ready_gate_binds_provisional_and_arm",
            "immutable_allocation_evidence_publication_protocol",
            "immutable_receipt_partial_write_is_fail_closed_and_preserved",
            "ready_gate_promotion_serialized_by_cli_state_lock",
            "mutable_watchdog_status_heartbeat_required",
            "watchdog_heartbeat_interval_seconds",
            "watchdog_heartbeat_max_age_seconds",
            "bound_mutation_lease_recheck_after_preprobe",
            "bound_mutation_timeouts_capped_to_remaining_lease",
            "offline_teardown_requires_mutable_watchdog_status",
            "normal_stop_emergency_lock_bypass_allowed",
            "absolute_lease_is_authority_cutoff_and_teardown_trigger_not_provider_absence_guarantee",
            "prearm_crash_orphan_window_disclosed",
        },
        "colab_cli": {
            "package_version", "distribution_name", "absolute_invocation_path",
            "resolved_entrypoint_path", "record_sha256", "record_file_closure",
            "record_unhashed_paths",
            "record_declared_hashes_verified_against_installed_bytes",
            "entrypoint_sha256", "entrypoint_byte_count", "source_files", "interpreter",
            "direct_requirement_strings", "direct_runtime_dependencies",
            "direct_dependency_record_hashes_verified",
            "direct_dependency_installed_file_bytes_verified_against_record",
            "direct_dependency_installed_file_bytes_trusted_local_tcb",
            "entire_tool_site_packages_directory_admitted_to_import_path",
            "entire_tool_site_packages_tree_bytes_authenticated",
            "transitive_and_unrelated_site_packages_bytes_trusted_mutable_local_tcb",
            "automatic_update_check_required", "fixed_state_root",
            "fixed_state_root_absolute_path",
            "bundled_oauth_config_path", "bundled_oauth_config_sha256",
            "bundled_oauth_config_byte_count",
            "client_oauth_config_argument_required",
            "ambient_home_oauth_config_read_allowed",
            "prepare_local_cli_state_operation_required",
            "token_log_history_contents_read_by_controller", "allocation_command",
            "restart_command", "actual_cli_launch",
            "assignment_probe_launch", "ambient_python_import_environment_allowed",
            "watchdog_controller_launch",
            "tool_site_packages_path", "isolated_python_flags",
            "system_site_initialization_allowed",
            "adjacent_bytecode_cache_loading_allowed",
            "bytecode_quarantine_path_template",
            "bytecode_quarantine_directory_mode_octal",
            "bytecode_quarantine_must_be_empty",
            "bytecode_quarantine_persistent_through_keepalive",
            "bootstrap_source_reauthenticated_before_every_child_launch",
            "target_parent_source_directory_inserted_by_bootstrap",
            "probe_imports_from_target_parent_allowed",
            "controller_contract_preloaded_from_authenticated_plan_input",
            "controller_contract_reauthenticated_by_bootstrap_before_execution",
            "cli_distribution_reauthenticated_after_preprobe_before_mutation",
            "same_user_midlaunch_runtime_mutation_excluded_claimed",
            "safe_keepalive_bootstrap_required",
            "keepalive_inherits_session_lock_descriptor",
            "stdlib_and_interpreter_shared_libraries_trusted_local_tcb",
        },
    }
    for field, expected in nested_keys.items():
        _require_exact_keys(plan.get(field), expected, f"execution-plan {field}")
    if (
        plan.get("schema_version") != EXECUTION_PLAN_SCHEMA
        or plan.get("semantics_version") != SEMANTICS_VERSION
        or canonical_sha256(plan) != plan.get("payload_sha256")
        or plan.get("status") != "candidate_frozen_no_execution_authority"
    ):
        raise ContractError("Phase-B execution-plan identity differs")
    model = plan.get("model")
    if not isinstance(model, Mapping) or (
        model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("file_closure_sha256") != MODEL_FILE_CLOSURE_SHA256
        or model.get("dtype") != "bfloat16"
        or model.get("files") != MODEL_FILES
        or model.get("snapshot_directory_name") != MODEL_REVISION
        or model.get("exact_file_set_required") is not True
        or model.get("proof_schema") != MODEL_SNAPSHOT_PROOF_SCHEMA
    ):
        raise ContractError("Phase-B model closure differs")
    feature = plan.get("feature")
    if not isinstance(feature, Mapping) or (
        feature.get("name") != FEATURE_NAME
        or feature.get("index") != FEATURE_INDEX
        or feature.get("activation_hook") != FEATURE_HOOK
        or feature.get("bundle_sha256") != FEATURE_BUNDLE_SHA256
        or feature.get("manifest_sha256") != FEATURE_MANIFEST_SHA256
        or feature.get("transcoder_revision") != TRANSCODER_REVISION
        or feature.get("activation_formula")
        != "relu(cuda_bfloat16_dot(encoder,state)+bfloat16_bias)"
        or feature.get("formatted_final_semantics")
        != "E8b-compatible rowwise BF16 dot at token_count_minus_one"
        or feature.get("all_token_semantics")
        != "vectorized BF16 matmul audited against rowwise final values"
    ):
        raise ContractError("Phase-B feature semantics differ")
    representation = plan.get("reference_directions")
    if not isinstance(representation, Mapping) or (
        representation.get("primary_hook") != PRIMARY_REFERENCE_HOOK
        or representation.get("secondary_hooks") != list(SECONDARY_REFERENCE_HOOKS)
        or representation.get("orientation") != "harmful_minus_benign"
        or representation.get("pair_balanced") is not True
        or representation.get("pair_count") != EXPECTED_REFERENCE_PAIR_COUNT
        or representation.get("reference_panel")
        != "fresh_reviewed_nonoverlapping_with_e9d"
        or representation.get("probe_coordinate_caveat")
        != "probe was trained post-final-RMSNorm; layer-31 cosines are coordinate proxies"
    ):
        raise ContractError("Phase-B reference-direction semantics differ")
    corpus = plan.get("corpus")
    if not isinstance(corpus, Mapping) or (
        corpus.get("row_count") != EXPECTED_ROW_COUNT
        or corpus.get("raw_prompt_text_transferred") is not False
        or corpus.get("pretokenized") is not True
        or corpus.get("no_padding") is not True
        or not _is_sha256(corpus.get("corpus_payload_sha256"))
        or not _is_sha256(corpus.get("manifest_payload_sha256"))
        or not _is_sha256(corpus.get("batch_schedule_payload_sha256"))
        or corpus.get("semantic_review_status")
        != "targeted_reviews_complete_execution_gate_closed_no_execution_authority"
    ):
        raise ContractError("Phase-B corpus envelope differs")
    batch = plan.get("batching")
    if not isinstance(batch, Mapping) or (
        batch.get("max_input_tokens") != MAX_INPUT_TOKENS
        or batch.get("max_batch_rows") != MAX_BATCH_ROWS
        or batch.get("max_batch_tokens") != MAX_BATCH_TOKENS
        or batch.get("equal_length_only") is not True
        or batch.get("padding_allowed") is not False
        or batch.get("truncation_allowed") is not False
        or batch.get("batch_size_one_fallback_on_failed_parity_audit") is not True
    ):
        raise ContractError("Phase-B batching contract differs")
    resource = plan.get("resource_contract")
    if not isinstance(resource, Mapping):
        raise ContractError("Phase-B resource contract is absent")
    required_resource = {
        "session_name": SESSION_NAME,
        "requested_accelerator": "A100",
        "maximum_gpu_sessions": 1,
        "maximum_authenticated_model_loads": 2,
        "minimum_cuda_memory_gib": 39.0,
        "hard_wall_seconds": 21600,
        "absolute_allocation_lease_seconds": 21600,
        "maximum_output_bytes": MAX_OUTPUT_BYTES,
        "generation_allowed": False,
        "intervention_allowed": False,
        "backward_allowed": False,
        "sealed_s10_confirmation_allowed": False,
    }
    if any(resource.get(key) != value for key, value in required_resource.items()):
        raise ContractError("Phase-B resource contract differs")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping) or (
        runtime.get("seed") != 9162026
        or runtime.get("direct_install_requirement_lines")
        != REMOTE_DIRECT_INSTALL_REQUIREMENT_LINES
        or runtime.get("post_install_top_level_package_versions")
        != PACKAGE_VERSIONS
        or runtime.get("full_installed_distribution_version_inventory_required")
        is not True
        or runtime.get("transitive_versions_preregistered") is not False
        or runtime.get("wheel_hashes_preregistered") is not False
        or runtime.get("python_implementation") != EXPECTED_PYTHON_IMPLEMENTATION
        or runtime.get("torch_cuda_version") != EXPECTED_TORCH_CUDA_VERSION
        or runtime.get("model_snapshot_proof_schema") != MODEL_SNAPSHOT_PROOF_SCHEMA
        or runtime.get("batch_parity_abs_tolerance") != 0.125
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("model_eval") is not True
        or runtime.get("gradients_enabled") is not False
        or runtime.get("allow_tf32") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("kernel_restart_after_install_required") is not True
        or runtime.get(
            "kernel_restart_after_private_upload_before_private_preflight_required"
        )
        is not True
        or runtime.get("premodel_remote_lifecycle_order")
        != PREMODEL_REMOTE_LIFECYCLE_ORDER
        or runtime.get("local_files_only_after_authenticated_snapshot") is not True
    ):
        raise ContractError("Phase-B runtime contract differs")
    audit_rows = runtime.get("audit_row_indices")
    if (
        not isinstance(audit_rows, list)
        or len(audit_rows) != 4
        or len(set(audit_rows)) != 4
        or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < EXPECTED_ROW_COUNT for index in audit_rows)
    ):
        raise ContractError("Phase-B parity-audit rows differ")
    accounting = plan.get("forward_accounting")
    if not isinstance(accounting, Mapping):
        raise ContractError("Phase-B forward accounting is absent")
    n = corpus["row_count"]
    total_tokens = corpus.get("total_token_count")
    batch_count = corpus.get("batch_count")
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens <= 0
        or total_tokens > n * MAX_INPUT_TOKENS
        or isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count <= 0
        or accounting.get("unique_scientific_prompt_equivalents") != n
        or accounting.get("unique_scientific_token_equivalents") != total_tokens
        or accounting.get("planned_scientific_invocations") != batch_count
        or accounting.get("planned_fallback_scientific_invocations") != n
        or accounting.get("maximum_scientific_invocations") != math.ceil(1.05 * n)
        or accounting.get("audit_invocations") != 7
        or accounting.get("audit_prompt_equivalents") != 10
        or isinstance(accounting.get("audit_token_equivalents"), bool)
        or not isinstance(accounting.get("audit_token_equivalents"), int)
        or accounting.get("audit_token_equivalents") <= 0
        or accounting.get("attempted_scientific_token_equivalent_ceiling")
        != math.ceil(1.05 * total_tokens)
        or accounting.get("attempted_total_token_equivalent_ceiling")
        != math.ceil(1.05 * total_tokens) + accounting.get("audit_token_equivalents")
        or accounting.get("attempted_total_invocation_ceiling")
        != math.ceil(1.05 * n) + 7
        or accounting.get("attempted_prompt_equivalent_ceiling")
        != math.ceil(1.05 * n) + 10
    ):
        raise ContractError("Phase-B forward-accounting closure differs")
    estimands = plan.get("estimands")
    required_estimands = [
        "formatted_final_e8_compatible",
        "last_user_content_token",
        "all_nonpadding_token_micro",
        "per_prompt_macro",
        "matched_final_activation_delta",
    ]
    if estimands != required_estimands:
        raise ContractError("Phase-B estimand closure differs")
    if plan.get("prespecified_contrasts") != [
        "row_balanced_position_summaries",
        "helpsteer_within_anchor_wrapper_minus_base_effects",
        "big_bench_valid_invalid_cells",
        "project_harmful_benign_splits",
    ]:
        raise ContractError("Phase-B prespecified contrast closure differs")
    bootstrap = plan.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or (
        bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("generator") != BOOTSTRAP_GENERATOR
        or bootstrap.get("unit") != "reference_pair"
        or bootstrap.get("lopo") is not True
        or bootstrap.get("helpsteer_anchor_replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("helpsteer_anchor_seed") != BOOTSTRAP_SEED
        or bootstrap.get("helpsteer_anchor_generator") != BOOTSTRAP_GENERATOR
        or bootstrap.get("helpsteer_familywise_alpha") != HELPSTEER_FAMILYWISE_ALPHA
        or bootstrap.get("helpsteer_familywise_method")
        != HELPSTEER_FAMILYWISE_METHOD
    ):
        raise ContractError("Phase-B bootstrap contract differs")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != REQUIRED_INPUT_ROLES:
        raise ContractError("Phase-B input closure is absent")
    for role, record in inputs.items():
        _require_exact_keys(record, {"path", "sha256", "byte_count", "sensitivity"}, f"input {role}")
        if not isinstance(role, str) or not isinstance(record, Mapping):
            raise ContractError("Phase-B input record is malformed")
        expected_sensitivity = (
            "private_tokenized"
            if role in {"batch_schedule", "token_bundle"}
            else "private_research_artifact"
            if role == "feature_bundle"
            else "public"
        )
        if (
            not isinstance(record.get("path"), str)
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("byte_count"), bool)
            or not isinstance(record.get("byte_count"), int)
            or record.get("byte_count") < 0
            or record.get("sensitivity") != expected_sensitivity
        ):
            raise ContractError(f"Phase-B input record differs: {role}")
    privacy = plan.get("privacy_and_transfer")
    if not isinstance(privacy, Mapping) or (
        privacy.get("raw_prompt_or_chat_text_uploaded") is not False
        or privacy.get("private_token_inputs") != ["batch_schedule", "token_bundle"]
        or privacy.get("private_research_artifact_inputs") != ["feature_bundle"]
        or privacy.get("raw_corpus_manifest_uploaded") is not False
        or privacy.get("semantic_review_queue_uploaded") is not False
        or privacy.get("colab_source_or_arguments_contain_private_tokens") is not False
        or privacy.get("private_inputs_uploaded_only_after_dependency_proof") is not True
        or privacy.get("second_stdlib_private_preflight_required") is not True
        or privacy.get("private_directory_mode_octal") != "0700"
        or privacy.get("private_file_mode_octal") != "0600"
    ):
        raise ContractError("Phase-B private-transfer contract differs")
    isolation = plan.get("session_isolation")
    if not isinstance(isolation, Mapping) or (
        isolation.get("fresh_named_session_required") is not True
        or isolation.get("q004_session_reuse_allowed") is not False
        or isolation.get("exact_endpoint_bound_before_private_upload") is not True
        or isolation.get("global_sync_or_prune_allowed") is not False
        or isolation.get("local_recomputation_before_exact_session_teardown") is not True
        or isolation.get("absolute_allocation_lease_seconds") != 21600
        or isolation.get("local_exact_assignment_watchdog_required") is not True
        or isolation.get("watchdog_abort_probe_timeout_seconds")
        != WATCHDOG_ABORT_PROBE_TIMEOUT_SECONDS
        or isolation.get("watchdog_abort_retry_interval_seconds")
        != WATCHDOG_ABORT_RETRY_INTERVAL_SECONDS
        or isolation.get("watchdog_abort_retry_until_absence_proven") is not True
        or isolation.get("watchdog_ambiguity_receipt_after_each_failed_attempt")
        is not True
        or isolation.get("watchdog_process_and_host_liveness_trusted_local_tcb")
        is not True
        or isolation.get("watchdog_survives_host_or_process_failure_claimed")
        is not False
        or isolation.get("emergency_abort_requires_scientific_completion") is not False
        or isolation.get("unattributed_server_orphan_cannot_be_excluded_by_local_name_probe")
        is not True
        or isolation.get("fixed_cli_state_lock_path") != CLI_STATE_LOCK_DISPLAY_PATH
        or isolation.get("cli_state_lock_protocol") != CLI_STATE_LOCK_PROTOCOL
        or isolation.get("cli_state_lock_acquire_timeout_seconds")
        != CLI_STATE_LOCK_ACQUIRE_TIMEOUT_SECONDS
        or isolation.get("cli_state_lock_poll_interval_seconds")
        != CLI_STATE_LOCK_POLL_INTERVAL_SECONDS
        or isolation.get("cli_state_lock_file_mode_octal") != "0600"
        or isolation.get("cli_state_lock_parent_mode_octal") != "0700"
        or isolation.get("session_name_probe_mutation_probe_lock_required") is not True
        or isolation.get("cooperating_s16_process_exclusivity_local_tcb") is not True
        or isolation.get("arbitrary_same_user_raw_cli_excluded_claimed") is not False
        or isolation.get("watchdog_exact_endpoint_abort_bounded_lock_then_bypass")
        is not True
        or isolation.get("watchdog_lock_bypass_may_race_inflight_cli_mutation") is not True
        or isolation.get("cli_state_lock_marker_authority_specific") is not False
        or isolation.get("lock_inherited_by_direct_child_until_exit") is not True
        or isolation.get("lock_holder_process_liveness_trusted_local_tcb") is not True
        or isolation.get("lock_survives_controller_and_direct_child_failure_claimed")
        is not False
        or isolation.get("provisional_allocation_is_scientific_authority") is not False
        or isolation.get("immutable_watchdog_arm_receipt_required") is not True
        or isolation.get("allocation_ready_gate_required_for_science") is not True
        or isolation.get("allocation_ready_gate_binds_provisional_and_arm") is not True
        or isolation.get("immutable_allocation_evidence_publication_protocol")
        != IMMUTABLE_JSON_PUBLICATION_PROTOCOL
        or isolation.get("immutable_receipt_partial_write_is_fail_closed_and_preserved")
        is not True
        or isolation.get("ready_gate_promotion_serialized_by_cli_state_lock") is not True
        or isolation.get("mutable_watchdog_status_heartbeat_required") is not True
        or isolation.get("watchdog_heartbeat_interval_seconds")
        != WATCHDOG_HEARTBEAT_INTERVAL_SECONDS
        or isolation.get("watchdog_heartbeat_max_age_seconds")
        != WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS
        or isolation.get("bound_mutation_lease_recheck_after_preprobe") is not True
        or isolation.get("bound_mutation_timeouts_capped_to_remaining_lease")
        is not True
        or isolation.get("offline_teardown_requires_mutable_watchdog_status")
        is not False
        or isolation.get("normal_stop_emergency_lock_bypass_allowed") is not False
        or isolation.get(
            "absolute_lease_is_authority_cutoff_and_teardown_trigger_not_provider_absence_guarantee"
        ) is not True
        or isolation.get("prearm_crash_orphan_window_disclosed") is not True
    ):
        raise ContractError("Phase-B session-isolation contract differs")
    cli = plan.get("colab_cli")
    if not isinstance(cli, Mapping) or (
        cli.get("package_version") != COLAB_CLI_VERSION
        or cli.get("distribution_name") != COLAB_CLI_DISTRIBUTION
        or cli.get("absolute_invocation_path") != COLAB_CLI_INVOCATION_PATH
        or cli.get("resolved_entrypoint_path")
        != COLAB_CLI_RESOLVED_ENTRYPOINT_PATH
        or cli.get("record_sha256") != COLAB_CLI_RECORD_SHA256
        or cli.get("record_file_closure") != COLAB_CLI_RECORD_FILE_CLOSURE
        or cli.get("record_unhashed_paths") != COLAB_CLI_RECORD_UNHASHED_PATHS
        or cli.get("record_declared_hashes_verified_against_installed_bytes") is not True
        or cli.get("entrypoint_sha256") != COLAB_CLI_ENTRYPOINT_SHA256
        or cli.get("entrypoint_byte_count") != COLAB_CLI_ENTRYPOINT_BYTE_COUNT
        or cli.get("source_files") != COLAB_CLI_SOURCE_FILES
        or cli.get("interpreter")
        != {
            "shebang_alias_path": COLAB_CLI_INTERPRETER_ALIAS_PATH,
            "resolved_executable_path": COLAB_CLI_INTERPRETER_RESOLVED_PATH,
            "implementation": COLAB_CLI_INTERPRETER_IMPLEMENTATION,
            "version": COLAB_CLI_INTERPRETER_VERSION,
            "executable_sha256": COLAB_CLI_INTERPRETER_SHA256,
            "executable_byte_count": COLAB_CLI_INTERPRETER_BYTE_COUNT,
        }
        or cli.get("direct_requirement_strings")
        != COLAB_CLI_DIRECT_REQUIREMENT_STRINGS
        or cli.get("direct_runtime_dependencies")
        != COLAB_CLI_DIRECT_RUNTIME_DEPENDENCIES
        or cli.get("direct_dependency_record_hashes_verified") is not True
        or cli.get("direct_dependency_installed_file_bytes_verified_against_record")
        is not False
        or cli.get("direct_dependency_installed_file_bytes_trusted_local_tcb")
        is not True
        or cli.get("entire_tool_site_packages_directory_admitted_to_import_path")
        is not True
        or cli.get("entire_tool_site_packages_tree_bytes_authenticated") is not False
        or cli.get(
            "transitive_and_unrelated_site_packages_bytes_trusted_mutable_local_tcb"
        )
        is not True
        or cli.get("automatic_update_check_required") is not False
        or cli.get("fixed_state_root") != "~/.config/colab-cli"
        or cli.get("fixed_state_root_absolute_path") != COLAB_CLI_STATE_ROOT_PATH
        or cli.get("bundled_oauth_config_path") != COLAB_CLI_OAUTH_CONFIG_PATH
        or cli.get("bundled_oauth_config_sha256")
        != COLAB_CLI_OAUTH_CONFIG_SHA256
        or cli.get("bundled_oauth_config_byte_count")
        != COLAB_CLI_OAUTH_CONFIG_BYTE_COUNT
        or cli.get("client_oauth_config_argument_required") is not True
        or cli.get("ambient_home_oauth_config_read_allowed") is not False
        or cli.get("prepare_local_cli_state_operation_required") is not True
        or cli.get("token_log_history_contents_read_by_controller") is not False
        or cli.get("allocation_command") != "new"
        or cli.get("restart_command") != "restart-kernel"
        or cli.get("actual_cli_launch")
        != "authenticated_interpreter_-I_-S_-B_-X_source_bootstrap"
        or cli.get("assignment_probe_launch")
        != "authenticated_interpreter_-I_-S_-B_-X_source_bootstrap"
        or cli.get("watchdog_controller_launch")
        != "authenticated_interpreter_-I_-S_-B_-X_source_bootstrap"
        or cli.get("ambient_python_import_environment_allowed") is not False
        or cli.get("tool_site_packages_path") != COLAB_CLI_SITE_PACKAGES_PATH
        or cli.get("isolated_python_flags") != ISOLATED_PYTHON_FLAGS
        or cli.get("system_site_initialization_allowed") is not False
        or cli.get("adjacent_bytecode_cache_loading_allowed") is not False
        or cli.get("bytecode_quarantine_path_template")
        != PYTHON_BYTECODE_QUARANTINE_DISPLAY_TEMPLATE
        or cli.get("bytecode_quarantine_directory_mode_octal") != "0700"
        or cli.get("bytecode_quarantine_must_be_empty") is not True
        or cli.get("bytecode_quarantine_persistent_through_keepalive") is not True
        or cli.get("bootstrap_source_reauthenticated_before_every_child_launch")
        is not True
        or cli.get("target_parent_source_directory_inserted_by_bootstrap") is not False
        or cli.get("probe_imports_from_target_parent_allowed") is not False
        or cli.get("controller_contract_preloaded_from_authenticated_plan_input")
        is not True
        or cli.get(
            "controller_contract_reauthenticated_by_bootstrap_before_execution"
        )
        is not True
        or cli.get("cli_distribution_reauthenticated_after_preprobe_before_mutation")
        is not True
        or cli.get("same_user_midlaunch_runtime_mutation_excluded_claimed") is not False
        or cli.get("safe_keepalive_bootstrap_required") is not True
        or cli.get("keepalive_inherits_session_lock_descriptor") is not False
        or cli.get("stdlib_and_interpreter_shared_libraries_trusted_local_tcb")
        is not True
    ):
        raise ContractError("Phase-B Colab CLI contract differs")


def authenticate_plan_inputs(
    plan: Mapping[str, Any], *, project_root: pathlib.Path
) -> dict[str, str]:
    validate_execution_plan(plan)
    root = project_root.resolve()
    hashes: dict[str, str] = {}
    for role, record in plan["inputs"].items():
        relative = pathlib.PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe Phase-B input path: {role}")
        path = root / pathlib.Path(*relative.parts)
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ContractError(f"Phase-B input escapes project: {role}") from error
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"Phase-B input missing or unsafe: {role}")
        require_no_symlink_components(root, path)
        digest = file_sha256(path)
        if digest != record["sha256"] or path.stat().st_size != record["byte_count"]:
            raise ContractError(f"Phase-B input bytes differ: {role}")
        if record.get("sensitivity") != "public":
            require_private_ancestry(root, path)
        hashes[role] = digest
    return hashes


def authenticate_public_plan_inputs_before_private_upload(
    plan: Mapping[str, Any], *, project_root: pathlib.Path
) -> dict[str, str]:
    """Authenticate the public plan-input subset and prove private paths absent."""

    validate_execution_plan(plan)
    root = project_root.resolve()
    hashes: dict[str, str] = {}
    private_roles: list[str] = []
    for role, record in plan["inputs"].items():
        relative = pathlib.PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe Phase-B input path: {role}")
        path = root / pathlib.Path(*relative.parts)
        require_no_symlink_components(root, path)
        if record.get("sensitivity") != "public":
            private_roles.append(role)
            if path.exists() or path.is_symlink():
                raise ContractError(
                    f"private Phase-B input present before dependency proof: {role}"
                )
            continue
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ContractError(f"Phase-B input escapes project: {role}") from error
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"Phase-B input missing or unsafe: {role}")
        digest = file_sha256(path)
        if digest != record["sha256"] or path.stat().st_size != record["byte_count"]:
            raise ContractError(f"Phase-B input bytes differ: {role}")
        hashes[role] = digest
    if sorted(private_roles) != ["batch_schedule", "feature_bundle", "token_bundle"]:
        raise ContractError("Phase-B private input role closure differs")
    return hashes


def validate_public_runtime_preflight(
    proof: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    public_input_hashes: Mapping[str, str],
    allocation: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    """Validate the stdlib predecessor for the dependency/runtime proof."""

    _require_exact_keys(proof, PUBLIC_RUNTIME_PROOF_KEYS, "public runtime preflight")
    expected_public_roles = {
        role
        for role, record in plan.get("inputs", {}).items()
        if record.get("sensitivity") == "public"
    }
    if (
        proof.get("schema_version") != RUNTIME_PROOF_SCHEMA
        or canonical_sha256(proof) != proof.get("payload_sha256")
        or proof.get("status")
        != "public_authority_and_inputs_authenticated_by_stdlib_before_dependency_install"
        or proof.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or proof.get("execution_request_payload_sha256")
        != request.get("payload_sha256")
        or proof.get("proposal_payload_sha256")
        != request.get("proposal_payload_sha256")
        or not _is_sha256(proof.get("allocation_provisional_receipt_payload_sha256"))
        or not _is_sha256(proof.get("allocation_ready_gate_payload_sha256"))
        or not _is_sha256(proof.get("watchdog_arm_receipt_payload_sha256"))
        or not _is_sha256(proof.get("session_binding_payload_sha256"))
        or set(public_input_hashes) != expected_public_roles
        or proof.get("uploaded_public_input_hashes") != dict(public_input_hashes)
        or proof.get("private_input_roles_confirmed_absent")
        != ["batch_schedule", "feature_bundle", "token_bundle"]
        or proof.get("scientific_modules_imported") is not False
        or proof.get("model_loaded") is not False
        or proof.get("model_forwards") != 0
    ):
        raise ContractError("Phase-B public runtime preflight differs")
    validate_allocation_evidence_bindings(
        proof, allocation, label="public runtime preflight"
    )
    if proof.get("session_binding_payload_sha256") != binding.get("payload_sha256"):
        raise ContractError("Phase-B public runtime preflight binding differs")


def validate_execution_request(
    request: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    validate_execution_plan(plan)
    _require_exact_keys(request, AUTHORIZED_REQUEST_KEYS, "authorized-request")
    if (
        request.get("schema_version") != EXECUTION_REQUEST_SCHEMA
        or canonical_sha256(request) != request.get("payload_sha256")
        or request.get("status") != "authorized_for_exact_phase_b_execution"
        or request.get("execution_plan_payload_sha256") != plan["payload_sha256"]
        or request.get("challenge") != authorization_challenge(plan)
        or not _is_sha256(request.get("proposal_payload_sha256"))
        or not _is_sha256(request.get("confirmation_text_sha256"))
        or request.get("confirmation_text_sha256")
        != hashlib.sha256(authorization_confirmation_text(plan).encode("utf-8")).hexdigest()
        or request.get("approval_reference")
        != "user-confirmation-sha256:" + request.get("confirmation_text_sha256", "")
    ):
        raise ContractError("Phase-B execution authority identity differs")
    required_true = (
        "researcher_approved",
        "receipt_grants_execution_authority",
        "private_token_upload_authorized",
        "private_research_artifact_upload_authorized",
        "a100_allocation_authorized",
        "target_model_load_authorized",
        "target_model_forward_authorized",
    )
    required_false = (
        "raw_prompt_upload_authorized",
        "intervention_authorized",
        "generation_authorized",
        "backward_authorized",
        "sealed_s10_confirmation_authorized",
        "phase_c_authorized",
        "phase_d_authorized",
    )
    if any(not _exact_bool(request.get(key), True) for key in required_true) or any(
        not _exact_bool(request.get(key), False) for key in required_false
    ):
        raise ContractError("Phase-B execution authority scope differs")


def validate_observed_distribution_inventory(inventory: Any) -> None:
    """Validate the deterministic, observed remote distribution/version listing.

    This is runtime evidence, not a preregistered transitive package closure.
    """

    if not isinstance(inventory, list) or not inventory:
        raise ContractError("Phase-B observed distribution inventory differs")
    normalized: list[tuple[str, str, str]] = []
    for record in inventory:
        if not isinstance(record, Mapping) or set(record) != {
            "canonical_name",
            "distribution_name",
            "version",
        }:
            raise ContractError("Phase-B observed distribution inventory differs")
        name = record.get("distribution_name")
        canonical = record.get("canonical_name")
        version = record.get("version")
        if (
            not isinstance(name, str)
            or _DISTRIBUTION_NAME.fullmatch(name) is None
            or not isinstance(canonical, str)
            or canonical != canonical_distribution_name(name)
            or not isinstance(version, str)
            or not version
            or len(version) > 255
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in version)
        ):
            raise ContractError("Phase-B observed distribution inventory differs")
        normalized.append((canonical, name, version))
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ContractError("Phase-B observed distribution inventory differs")
    for distribution_name, expected_version in PACKAGE_VERSIONS.items():
        canonical = canonical_distribution_name(distribution_name)
        matches = [record for record in normalized if record[0] == canonical]
        if len(matches) != 1 or matches[0][2] != expected_version:
            raise ContractError("Phase-B observed distribution inventory differs")


def validate_dependency_runtime_proof(
    proof: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    public_preflight: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        proof, DEPENDENCY_RUNTIME_PROOF_KEYS, "dependency runtime proof"
    )
    parse_utc(proof.get("created_at_utc"))
    if (
        public_preflight.get("schema_version") != RUNTIME_PROOF_SCHEMA
        or canonical_sha256(public_preflight)
        != public_preflight.get("payload_sha256")
        or proof.get("schema_version") != DEPENDENCY_RUNTIME_PROOF_SCHEMA
        or canonical_sha256(proof) != proof.get("payload_sha256")
        or proof.get("status")
        != "five_direct_requirements_six_top_level_versions_and_a100_verified_before_model_load"
        or proof.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or proof.get("execution_request_payload_sha256")
        != request.get("payload_sha256")
        or proof.get("proposal_payload_sha256")
        != request.get("proposal_payload_sha256")
        or proof.get("allocation_provisional_receipt_payload_sha256")
        != public_preflight.get("allocation_provisional_receipt_payload_sha256")
        or proof.get("allocation_ready_gate_payload_sha256")
        != public_preflight.get("allocation_ready_gate_payload_sha256")
        or proof.get("watchdog_arm_receipt_payload_sha256")
        != public_preflight.get("watchdog_arm_receipt_payload_sha256")
        or proof.get("session_binding_payload_sha256")
        != public_preflight.get("session_binding_payload_sha256")
        or proof.get("public_preflight_payload_sha256")
        != public_preflight.get("payload_sha256")
        or proof.get("fresh_kernel_after_dependency_install") is not True
        or not isinstance(proof.get("python"), str)
        or not proof.get("python")
        or proof.get("python_implementation") != EXPECTED_PYTHON_IMPLEMENTATION
        or not isinstance(proof.get("platform"), str)
        or not proof.get("platform")
        or proof.get("torch_cuda") != EXPECTED_TORCH_CUDA_VERSION
        or proof.get("direct_install_requirement_lines")
        != REMOTE_DIRECT_INSTALL_REQUIREMENT_LINES
        or proof.get("top_level_package_versions") != PACKAGE_VERSIONS
        or not isinstance(proof.get("installed_distribution_inventory"), list)
        or proof.get("installed_distribution_count")
        != len(proof.get("installed_distribution_inventory", []))
        or proof.get("transitive_versions_status") != "observed_not_preregistered"
        or proof.get("wheel_hashes_status") != "not_captured_not_preregistered"
        or proof.get("scientific_imports_completed") != SCIENTIFIC_IMPORT_PROOF
        or proof.get("device_capability") != [8, 0]
        or not isinstance(proof.get("device"), str)
        or "A100" not in proof.get("device", "")
        or not isinstance(proof.get("cuda_total_memory_gib"), (int, float))
        or isinstance(proof.get("cuda_total_memory_gib"), bool)
        or proof.get("cuda_total_memory_gib")
        < plan.get("resource_contract", {}).get("minimum_cuda_memory_gib", math.inf)
        or proof.get("cublas_workspace_config") != ":4096:8"
        or proof.get("cudnn_benchmark") is not False
        or proof.get("allow_tf32") is not False
        or proof.get("deterministic_algorithms") is not True
        or proof.get("model_loaded") is not False
        or proof.get("model_forwards") != 0
    ):
        raise ContractError("Phase-B dependency runtime proof differs")
    validate_observed_distribution_inventory(
        proof.get("installed_distribution_inventory")
    )


def validate_private_runtime_proof(
    proof: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    public_preflight: Mapping[str, Any],
    dependency_proof: Mapping[str, Any],
    private_input_hashes: Mapping[str, str],
    allocation: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    """Validate the exact public-safe proof emitted after the private restart."""

    _require_exact_keys(proof, PRIVATE_RUNTIME_PROOF_KEYS, "private runtime proof")
    parse_utc(proof.get("created_at_utc"))
    expected_roles = sorted(
        role
        for role, record in plan.get("inputs", {}).items()
        if record.get("sensitivity") != "public"
    )
    if (
        expected_roles != ["batch_schedule", "feature_bundle", "token_bundle"]
        or set(private_input_hashes) != set(expected_roles)
        or any(not _is_sha256(value) for value in private_input_hashes.values())
        or proof.get("schema_version") != PRIVATE_RUNTIME_PROOF_SCHEMA
        or canonical_sha256(proof) != proof.get("payload_sha256")
        or proof.get("status")
        != "three_private_inputs_authenticated_by_stdlib_after_dependency_proof"
        or proof.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or proof.get("execution_request_payload_sha256")
        != request.get("payload_sha256")
        or proof.get("proposal_payload_sha256")
        != request.get("proposal_payload_sha256")
        or proof.get("session_binding_payload_sha256")
        != binding.get("payload_sha256")
        or proof.get("uploaded_private_input_hashes") != dict(private_input_hashes)
        or proof.get("private_input_roles") != expected_roles
        or proof.get("public_preflight_payload_sha256")
        != public_preflight.get("payload_sha256")
        or proof.get("dependency_proof_payload_sha256")
        != dependency_proof.get("payload_sha256")
        or proof.get("scientific_modules_imported_by_this_process") is not False
        or proof.get("model_loaded") is not False
        or proof.get("model_forwards") != 0
    ):
        raise ContractError("Phase-B private runtime proof differs")
    validate_dependency_runtime_proof(
        dependency_proof,
        plan=plan,
        request=request,
        public_preflight=public_preflight,
    )
    validate_allocation_evidence_bindings(
        proof, allocation, label="private runtime proof"
    )


def validate_runtime_proof_key_closures(
    public_preflight: Mapping[str, Any],
    dependency_proof: Mapping[str, Any],
    private_proof: Mapping[str, Any],
) -> None:
    """Permit packagers without authority inputs to reject public-field drift."""

    _require_exact_keys(
        public_preflight,
        PUBLIC_RUNTIME_PROOF_KEYS,
        "public runtime preflight",
    )
    _require_exact_keys(
        dependency_proof,
        DEPENDENCY_RUNTIME_PROOF_KEYS,
        "dependency runtime proof",
    )
    _require_exact_keys(private_proof, PRIVATE_RUNTIME_PROOF_KEYS, "private runtime proof")


def validate_premodel_proof_bindings(
    running: Mapping[str, Any],
    *,
    public_preflight: Mapping[str, Any],
    dependency_proof: Mapping[str, Any],
    private_proof: Mapping[str, Any],
) -> None:
    bindings = running.get("premodel_proofs")
    _require_exact_keys(bindings, PREMODEL_PROOF_BINDING_KEYS, "premodel proof bindings")
    expected = {
        "preinstall_payload_sha256": public_preflight.get("payload_sha256"),
        "dependency_payload_sha256": dependency_proof.get("payload_sha256"),
        "private_preflight_payload_sha256": private_proof.get("payload_sha256"),
    }
    if (
        bindings != expected
        or any(not _is_sha256(value) for value in expected.values())
    ):
        raise ContractError("Phase-B premodel proof bindings differ")


def validate_no_authority_proposal(
    request: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    _require_exact_keys(request, NO_AUTHORITY_PROPOSAL_KEYS, "no-authority proposal")
    if (
        request.get("schema_version") != EXECUTION_REQUEST_SCHEMA
        or canonical_sha256(request) != request.get("payload_sha256")
        or request.get("status") != "proposal_only_no_execution_authority"
        or request.get("execution_plan_payload_sha256") != plan.get("payload_sha256")
        or request.get("challenge") != authorization_challenge(plan)
        or request.get("researcher_approved") is not False
        or request.get("receipt_grants_execution_authority") is not False
        or request.get("requested_scope") != PROPOSAL_REQUESTED_SCOPE
    ):
        raise ContractError("Phase-B no-authority proposal differs")
    authority_flags = (
        "private_token_upload_authorized",
        "private_research_artifact_upload_authorized",
        "a100_allocation_authorized",
        "target_model_load_authorized",
        "target_model_forward_authorized",
        "raw_prompt_upload_authorized",
        "intervention_authorized",
        "generation_authorized",
        "backward_authorized",
        "sealed_s10_confirmation_authorized",
        "phase_c_authorized",
        "phase_d_authorized",
    )
    if any(request.get(key) is not False for key in authority_flags):
        raise ContractError("Phase-B proposal contains authority")
    required_text = authorization_confirmation_text(plan)
    if (
        request.get("required_confirmation_text") != required_text
        or request.get("required_confirmation_text_sha256")
        != hashlib.sha256(required_text.encode("utf-8")).hexdigest()
    ):
        raise ContractError("Phase-B proposal confirmation text differs")


def authorization_challenge(plan: Mapping[str, Any]) -> str:
    payload_hash = plan.get("payload_sha256")
    if not _is_sha256(payload_hash):
        raise ContractError("Phase-B plan hash is unavailable for challenge")
    return hashlib.sha256(
        ("s16-phase-b-exact-authority-challenge/1:" + payload_hash).encode("ascii")
    ).hexdigest()


def authorization_confirmation_text(plan: Mapping[str, Any]) -> str:
    """Return the exact public-safe sentence the researcher must confirm."""

    payload_hash = plan.get("payload_sha256")
    if not _is_sha256(payload_hash):
        raise ContractError("Phase-B plan hash is unavailable for confirmation text")
    accounting = plan.get("forward_accounting", {})
    corpus = plan.get("corpus", {})
    return (
        "I authorize the S16 controller to contact Google Colab and execute exactly "
        f"execution plan {payload_hash}: prepare the fixed ~/.config/colab-cli state "
        "by preserving settings while disabling automatic update checks and setting "
        "private modes without reading token, log, or history contents; use only the "
        "authenticated bundled google-colab-cli OAuth client config, never the ambient "
        "~/.colab-cli-oauth-config.json fallback; allocate "
        f"exactly one A100 assignment named {SESSION_NAME}. The allocation is only "
        "provisional until an immutable watchdog-arm receipt and an immutable "
        "allocation-ready gate bind that receipt; no scientific descendant may "
        "accept the provisional receipt alone. Then upload the public plan, "
        "authority records, control receipts, and public inputs and authenticate them "
        "with a stdlib-only preinstall proof; install exactly the five directly pinned "
        "requirement lines "
        "and, after a first exact-session restart, record the full installed "
        "distribution-version inventory "
        "and require the six preregistered top-level package versions plus torch/CUDA "
        "checks, with transitive versions observed rather than preregistered and wheel "
        "hashes neither captured nor preregistered; only then upload exactly the private "
        "batch schedule, private token bundle, and authenticated S16-local private "
        "feature bundle, restart the exact session a second time after all three private "
        "uploads, and authenticate their hashes and modes with a second stdlib-only "
        "proof; acquire/download and "
        f"authenticate Qwen/Qwen3-4B revision {MODEL_REVISION}; and load it and run "
        "the authorized forward-only activation-census passes. I authorize at most "
        f"{accounting.get('maximum_scientific_invocations')} scientific invocations, "
        f"{accounting.get('attempted_scientific_token_equivalent_ceiling')} scientific "
        f"token-equivalents, {accounting.get('audit_invocations')} audit invocations, "
        f"{accounting.get('audit_prompt_equivalents')} audit prompt-equivalents, and "
        f"{plan.get('resource_contract', {}).get('maximum_authenticated_model_loads')} "
        f"authenticated model loads over {corpus.get('row_count')} unique corpus rows. "
        "I do not authorize raw prompt upload, generation, intervention, backward "
        "passes, sealed S10 confirmation access, Phase C, or Phase D. The exact "
        "assignment may be aborted independently of scientific completion, download, "
        "or local recomputation and is governed by one 21600-second absolute lease "
        "from allocation through teardown. At its deadline, the watchdog may retry "
        "abort of only that exact endpoint until authenticated absence is proven. "
        "The 21600-second deadline is an authority cutoff and teardown trigger, not "
        "a guarantee of provider-confirmed absence. "
        "Every S16 session-name probe and mutation uses an authenticated private "
        "POSIX advisory lock spanning pre-probe, mutation, and post-probe, and every "
        "CLI or assignment-probe child uses the authenticated CLI interpreter in "
        "-I -S -B isolated mode with a sanitized import environment and a persistent "
        "private empty bytecode-cache quarantine. The bootstrap does not add either "
        "probe or controller target-parent directory to sys.path; it reauthenticates "
        "and preloads the plan-bound controller contract bytes, while the probe resolves "
        "colab_cli only from the fixed admitted tool site-packages tree. I understand "
        "that this bootstrap admits "
        "the entire fixed tool site-packages directory: only the google-colab-cli "
        "RECORD-declared bytes are authenticated against that RECORD, while direct-"
        "dependency RECORD metadata is checked but the full direct, transitive, and "
        "unrelated installed package bytes are not. Those bytes, the standard library, "
        "dynamic loader, linked libraries, and same-user mutation between authentication "
        "and execution remain mutable local "
        "trusted-computing-base limits rather than a fully frozen runtime. I understand this lock "
        "excludes cooperating same-user S16 processes only, not arbitrary raw CLI "
        "or other uncooperative same-user state mutation. The authority-neutral marker "
        "is authenticated separately from each operation's plan and request; the lock "
        "file descriptor is inherited by the direct CLI/probe child until it exits; "
        "the authenticated CLI's keep-alive launcher closes nonstandard descriptors, "
        "so the lock does not enter that descendant. Simultaneous controller/direct-child "
        "failure remains outside the claim. At lease expiry, exact-endpoint "
        "watchdog abort tries the lock for a bounded interval and may bypass it rather "
        "than wait indefinitely; that emergency bypass can race an in-flight CLI mutation. "
        "I understand that a CLI crash or timeout after colab new but before "
        "authenticated endpoint capture and watchdog arming can leave an "
        "unattributed server-side assignment outside exact-endpoint teardown; the "
        "local name probe cannot exclude that residual orphan window."
        " The authenticated arm receipt is durable evidence, not a durable-service "
        "guarantee: continued watchdog-process and host liveness remain a trusted "
        "local TCB and are not claimed to survive process or host failure."
    )


def validate_corpus_manifest(manifest: Mapping[str, Any]) -> None:
    review_gate = manifest.get("hard_pre_execution_review_gate")
    nonoverlap = manifest.get("reference_panel_nonoverlap")
    tokenizer = manifest.get("tokenizer")
    token_policy = manifest.get("token_policy")
    if (
        manifest.get("schema_version") != CORPUS_SCHEMA
        or canonical_sha256(manifest) != manifest.get("payload_sha256")
    ):
        raise ContractError("Phase-B corpus authentication differs")
    if (
        manifest.get("execution_ready") is not True
        or manifest.get("semantic_review_status")
        != "targeted_reviews_complete_execution_gate_closed_no_execution_authority"
        or not isinstance(review_gate, Mapping)
        or review_gate.get("status") != "closed"
        or review_gate.get("selected_helpsteer_anchor_count") != 900
        or review_gate.get("selected_helpsteer_reviewed_benign_count") != 900
        or review_gate.get("selected_helpsteer_pending_count") != 0
        or review_gate.get(
            "all_900_selected_helpsteer_anchors_semantically_accepted_benign"
        )
        is not True
        or review_gate.get(
            "fresh_reference_pairs_independent_second_review_complete"
        )
        is not True
    ):
        raise ContractError("Phase-B targeted semantic review gate is not closed")
    if (
        manifest.get("status")
        != "candidate_materialized_machine_authenticated_targeted_reviews_complete_no_execution_authority"
        or manifest.get("row_count") != EXPECTED_ROW_COUNT
        or manifest.get("reference_pair_count") != EXPECTED_REFERENCE_PAIR_COUNT
        or manifest.get("raw_prompt_text_in_manifest") is not False
        or manifest.get("chat_text_in_manifest") is not False
        or manifest.get("input_ids_in_manifest") is not False
        or manifest.get("assistant_response_fields_used") is not False
        or manifest.get("sealed_s10_confirmation_included") is not False
        or manifest.get("execution_authority") is not False
        or not _is_sha256(manifest.get("corpus_payload_sha256"))
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("model_id") != MODEL_ID
        or tokenizer.get("model_revision") != MODEL_REVISION
        or tokenizer.get("addressable_token_count") != 151669
        or not isinstance(token_policy, Mapping)
        or token_policy.get("maximum_full_chat_tokens") != MAX_INPUT_TOKENS
        or token_policy.get("truncation_allowed") is not False
        or token_policy.get("padding_used_in_bundle") is not False
        or not isinstance(token_policy.get("maximum_observed_chat_tokens"), int)
        or token_policy.get("maximum_observed_chat_tokens") > MAX_INPUT_TOKENS
    ):
        raise ContractError("Phase-B corpus manifest differs")
    if (
        not isinstance(nonoverlap, Mapping)
        or nonoverlap.get("historical_panel")
        != "s9_e9b_prompt_panel_consumed_by_e9d"
        or nonoverlap.get("historical_prompt_count") != 24
        or nonoverlap.get("fresh_prompt_count") != 24
        or nonoverlap.get("raw_sha256_overlap_count") != 0
        or nonoverlap.get("canonical_exact_overlap_count") != 0
        or nonoverlap.get("historical_rows_in_corpus") != 0
    ):
        raise ContractError("Phase-B fresh reference-panel nonoverlap differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        raise ContractError("Phase-B corpus row closure differs")
    row_indices: set[int] = set()
    row_ids: set[str] = set()
    pairs: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("Phase-B corpus row is malformed")
        index, row_id, length = row.get("row_index"), row.get("row_id"), row.get("token_count")
        user_start, user_end = row.get("user_token_start"), row.get("user_token_end_exclusive")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(row_id, str)
            or not row_id
            or isinstance(length, bool)
            or not isinstance(length, int)
            or not 1 <= length <= MAX_INPUT_TOKENS
            or not isinstance(row.get("category"), str)
            or not row.get("category")
            or not _is_sha256(row.get("input_ids_sha256"))
            or isinstance(user_start, bool)
            or not isinstance(user_start, int)
            or isinstance(user_end, bool)
            or not isinstance(user_end, int)
            or not 0 <= user_start < user_end <= length
            or row.get("final_position_index") != length - 1
        ):
            raise ContractError("Phase-B corpus row fields differ")
        row_indices.add(index)
        row_ids.add(row_id)
        pair_id, role = row.get("reference_pair_id"), row.get("reference_member")
        if pair_id is None and role is None:
            continue
        if not isinstance(pair_id, str) or role not in {"harmful", "benign"}:
            raise ContractError("Phase-B reference-pair metadata differs")
        if row.get("risk_class") != role:
            raise ContractError("Phase-B reference-pair risk/member differs")
        if row.get("source_id") != "s16_fresh_reference_pairs":
            raise ContractError("Phase-B reference row is not from the fresh panel")
        pairs.setdefault(pair_id, []).append(role)
    if row_indices != set(range(EXPECTED_ROW_COUNT)) or len(row_ids) != EXPECTED_ROW_COUNT:
        raise ContractError("Phase-B corpus indices or identities differ")
    if len(pairs) != EXPECTED_REFERENCE_PAIR_COUNT or any(
        sorted(roles) != ["benign", "harmful"] for roles in pairs.values()
    ):
        raise ContractError("Phase-B reference-pair closure differs")


def validate_batch_schedule(
    schedule: Mapping[str, Any], *, corpus: Mapping[str, Any]
) -> None:
    validate_corpus_manifest(corpus)
    if (
        schedule.get("schema_version") != BATCH_SCHEDULE_SCHEMA
        or canonical_sha256(schedule) != schedule.get("payload_sha256")
        or schedule.get("status") != "authenticated_equal_length_schedule_execution_not_authorized"
        or schedule.get("corpus_payload_sha256")
        != corpus.get("corpus_payload_sha256")
        or schedule.get("row_count") != EXPECTED_ROW_COUNT
        or schedule.get("padding_allowed") is not False
        or schedule.get("truncation_allowed") is not False
        or schedule.get("exact_length_only") is not True
        or schedule.get("execution_authority") is not False
        or schedule.get("maximum_rows_per_batch") != MAX_BATCH_ROWS
        or schedule.get("maximum_tokens_per_batch") != MAX_BATCH_TOKENS
    ):
        raise ContractError("Phase-B batch-schedule identity differs")
    lengths = {row["row_index"]: row["token_count"] for row in corpus["rows"]}
    seen: list[int] = []
    batches = schedule.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ContractError("Phase-B batch schedule is empty")
    for expected_id, batch in enumerate(batches):
        if (
            not isinstance(batch, Mapping)
            or batch.get("batch_index") != expected_id
            or batch.get("batch_id") != f"s16b-batch-{expected_id:04d}"
        ):
            raise ContractError("Phase-B batch IDs differ")
        indices = batch.get("row_indices")
        token_count = batch.get("token_count")
        if (
            not isinstance(indices, list)
            or not indices
            or len(indices) > MAX_BATCH_ROWS
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 1
            or token_count > MAX_INPUT_TOKENS
            or any(index not in lengths or lengths[index] != token_count for index in indices)
            or len(indices) * token_count > MAX_BATCH_TOKENS
            or batch.get("row_count") != len(indices)
            or batch.get("total_tokens") != len(indices) * token_count
            or batch.get("row_ids")
            != [corpus["rows"][index]["row_id"] for index in indices]
        ):
            raise ContractError("Phase-B equal-length batch differs")
        seen.extend(indices)
    expected_order = sorted(range(EXPECTED_ROW_COUNT), key=lambda index: (lengths[index], index))
    if seen != expected_order:
        raise ContractError(
            "Phase-B schedule must cover rows once in deterministic length/index order"
        )
    total = sum(lengths.values())
    if (
        schedule.get("batch_count") != len(batches)
        or schedule.get("total_token_count") != total
        or schedule.get("maximum_observed_token_count") != max(lengths.values())
    ):
        raise ContractError("Phase-B schedule totals differ")


def _numpy():
    # Keep controller/watchdog imports stdlib-only; numerical workers load
    # NumPy only when one of the scientific helpers is actually called.
    import numpy

    return numpy


def quantiles(values: Sequence[float], probabilities: Sequence[float]) -> dict[str, float]:
    np = _numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ContractError("quantile input must be a nonempty finite vector")
    result = np.quantile(array, np.asarray(probabilities), method="linear")
    return {format(float(p), ".6g"): float(v) for p, v in zip(probabilities, result)}


def historical_percentile_interval(values: Sequence[float]) -> list[float]:
    np = _numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ContractError("percentile input must be a nonempty finite vector")
    return [
        float(np.count_nonzero(array < HISTORICAL_DOSE) / array.size),
        float(np.count_nonzero(array <= HISTORICAL_DOSE) / array.size),
    ]


def pair_balanced_direction(
    harmful: np.ndarray, benign: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    np = _numpy()
    h = np.asarray(harmful, dtype=np.float32)
    b = np.asarray(benign, dtype=np.float32)
    if h.shape != b.shape or h.ndim != 2 or h.shape[0] != EXPECTED_REFERENCE_PAIR_COUNT:
        raise ContractError("reference residual matrices differ")
    pair_deltas = h.astype(np.float64) - b.astype(np.float64)
    mean = pair_deltas.mean(axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ContractError("reference mean-difference direction is degenerate")
    return (mean / norm).astype(np.float32), pair_deltas, norm


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    np = _numpy()
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ContractError("cosine vectors differ")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def bootstrap_cosine_interval(
    pair_deltas: np.ndarray,
    comparator: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    np = _numpy()
    deltas = np.asarray(pair_deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[0] != EXPECTED_REFERENCE_PAIR_COUNT:
        raise ContractError("bootstrap pair-delta closure differs")
    rng = np.random.Generator(np.random.PCG64(seed))
    values: list[float] = []
    comp = np.asarray(comparator, dtype=np.float64).reshape(-1)
    comp_norm = float(np.linalg.norm(comp))
    if comp.shape != (deltas.shape[1],) or comp_norm == 0.0:
        raise ContractError("bootstrap comparator differs")
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(
            0, deltas.shape[0], size=(count, deltas.shape[0])
        )
        samples = deltas[indices].mean(axis=1)
        norms = np.linalg.norm(samples, axis=1) * comp_norm
        if np.any(norms == 0.0):
            raise ContractError("bootstrap produced a degenerate direction")
        values.extend(((samples @ comp) / norms).tolist())
    if len(values) != replicates:
        raise ContractError("bootstrap produced a degenerate direction")
    return {
        "replicates": replicates,
        "seed": seed,
        "generator": BOOTSTRAP_GENERATOR,
        "unit": "reference_pair",
        "interval_95": [
            float(np.quantile(values, 0.025, method="linear")),
            float(np.quantile(values, 0.975, method="linear")),
        ],
    }


def bootstrap_mean_interval(
    pair_values: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Pair-bootstrap a scalar mean with the frozen PCG64 stream."""

    np = _numpy()

    values = np.asarray(pair_values, dtype=np.float64)
    if (
        values.shape != (EXPECTED_REFERENCE_PAIR_COUNT,)
        or not np.isfinite(values).all()
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ContractError("scalar pair-bootstrap closure differs")
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled_means: list[float] = []
    for start in range(0, replicates, 1024):
        count = min(1024, replicates - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        sampled_means.extend(values[indices].mean(axis=1).tolist())
    return {
        "replicates": replicates,
        "seed": seed,
        "generator": BOOTSTRAP_GENERATOR,
        "unit": "reference_pair",
        "estimand": "mean_harmful_minus_benign_formatted_final_activation",
        "interval_95": [
            float(np.quantile(sampled_means, 0.025, method="linear")),
            float(np.quantile(sampled_means, 0.975, method="linear")),
        ],
    }


def helpsteer_familywise_mean_intervals(
    wrapper_minus_base: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap three matched-anchor means with frozen familywise coverage."""

    np = _numpy()

    values = np.asarray(wrapper_minus_base, dtype=np.float64)
    if (
        values.shape != (HELPSTEER_ANCHOR_COUNT, len(HELPSTEER_WRAPPER_VARIANTS))
        or not np.isfinite(values).all()
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ContractError("HelpSteer paired-anchor bootstrap closure differs")
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        indices = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        sampled[start : start + count] = values[indices].mean(axis=1)
    tail = HELPSTEER_FAMILYWISE_ALPHA / (2.0 * values.shape[1])
    return {
        "replicates": replicates,
        "seed": seed,
        "generator": BOOTSTRAP_GENERATOR,
        "unit": "matched_helpsteer_anchor_family",
        "anchor_count": HELPSTEER_ANCHOR_COUNT,
        "familywise_confidence": 0.95,
        "method": HELPSTEER_FAMILYWISE_METHOD,
        "two_sided_tail_probability_per_contrast": tail,
        "intervals": {
            variant: [
                float(np.quantile(sampled[:, index], tail, method="linear")),
                float(
                    np.quantile(sampled[:, index], 1.0 - tail, method="linear")
                ),
            ]
            for index, variant in enumerate(HELPSTEER_WRAPPER_VARIANTS)
        },
    }


def phase_b_prioritization(
    *,
    helpsteer_familywise: Mapping[str, Any],
    harmful_minus_benign_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen descriptive A/B/C prioritization rule."""

    intervals = helpsteer_familywise.get("intervals")
    pair_interval = harmful_minus_benign_bootstrap.get("interval_95")
    if (
        not isinstance(intervals, Mapping)
        or set(intervals) != set(HELPSTEER_WRAPPER_VARIANTS)
        or any(
            not isinstance(intervals[variant], Sequence)
            or len(intervals[variant]) != 2
            or not all(math.isfinite(float(value)) for value in intervals[variant])
            for variant in HELPSTEER_WRAPPER_VARIANTS
        )
        or not isinstance(pair_interval, Sequence)
        or len(pair_interval) != 2
        or not all(math.isfinite(float(value)) for value in pair_interval)
    ):
        raise ContractError("Phase-B prioritization evidence differs")
    positive = {
        variant: float(intervals[variant][0]) > 0.0
        for variant in HELPSTEER_WRAPPER_VARIANTS
    }
    nonpositive = {
        variant: float(intervals[variant][1]) <= 0.0
        for variant in HELPSTEER_WRAPPER_VARIANTS
    }
    category_a = sum(positive.values()) >= 2
    harmful_delta = float(pair_interval[0]) > 0.0
    category_b = harmful_delta and all(nonpositive.values())
    return {
        "category": (
            "A_benign_stance_signal_detected"
            if category_a
            else "B_reference_pair_positive_without_detected_helpsteer_wrapper_signal"
            if category_b
            else "C_mixed_or_indeterminate"
        ),
        "category_a_components": {
            "helpsteer_positive_lower_bound_by_wrapper": positive,
            "positive_wrapper_count": sum(positive.values()),
            "required_count": 2,
            "predicate_met": category_a,
        },
        "category_b_components": {
            "harmful_minus_benign_final_lower_bound_positive": harmful_delta,
            "helpsteer_nonpositive_upper_bound_by_wrapper": nonpositive,
            "all_helpsteer_upper_bounds_nonpositive": all(nonpositive.values()),
            "predicate_met": category_b,
        },
        "evidence_boundary": (
            "descriptive pre-execution prioritization only; not a feature-selectivity "
            "or semantic-function claim and never Phase-C execution authority"
        ),
        "phase_c_authorized": False,
    }


def lopo_cosines(pair_deltas: np.ndarray, comparator: np.ndarray) -> list[float | None]:
    np = _numpy()
    deltas = np.asarray(pair_deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[0] != EXPECTED_REFERENCE_PAIR_COUNT:
        raise ContractError("LOPO pair-delta closure differs")
    return [
        cosine(np.delete(deltas, index, axis=0).mean(axis=0), comparator)
        for index in range(deltas.shape[0])
    ]
