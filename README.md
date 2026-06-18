# Handwritten Chemistry Lab Notebook Parser

Converts a scanned/photographed handwritten chemistry lab notebook page into
structured, machine-readable JSON, using a hosted **multimodal LLM (OpenAI API)**
as the recognition + reasoning engine, driven by a **multi-pass, schema-constrained**
pipeline. See [`spec.md`](spec.md) for the full design.

## Why an LLM engine

The earlier version used classical CV + heuristic OCR. On real handwriting it
produced nonsense — there was no handwriting model, so the transcript was empty
and every downstream stage degraded. This version sends the page image to a
vision-capable model and constrains every response with **Structured Outputs**, so
the model reads the handwriting, recovers symbols, infers SMILES for drawn
molecules, and summarizes the experiment — while still emitting the exact JSON
schema below.

Inference runs on OpenAI's servers, so **a local GPU is not required**.

## Architecture

Multi-pass pipeline (§3, §7), each pass swappable and schema-constrained:

```
condition image → transcription → chemistry → experiment → assemble → confidence → validate
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Image conditioning | `imaging.py` | load, optional deskew, downscale, base64 encode |
| Transcription pass (vision) | `passes/transcription.py` | layout regions, line transcript, recovered symbols |
| Chemistry pass (vision) | `passes/chemistry.py` | reagents, formulas, drawn structures (SMILES), concentrations |
| Experiment pass (text-only) | `passes/experiment.py` | goal/conditions/procedure/observations/results, evidence-linked |
| Engine | `llm/` | `LLMEngine` protocol; `OpenAIEngine` (vision + Structured Outputs) and offline `StubEngine` |
| Confidence + schema | `validation/` | per-field + overall confidence, review flags, Pydantic schema enforcement |

The experiment pass receives **only the transcript + chemistry** (no pixels) and
must cite transcript line ids, minimizing hallucination (§10). RDKit canonicalizes
SMILES when installed.

## Requirements

- An OpenAI API key in `OPENAI_API_KEY`.
- Optionally `OPENAI_MODEL` to pick the model (defaults to `gpt-4o`; must be a
  vision-capable model that supports Structured Outputs).

## Installation

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell reports "running scripts is disabled on this system", allow it for
> the current session only, then activate again:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> (Alternatively use cmd.exe: `python -m venv .venv` then `.venv\Scripts\activate.bat`.)

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Chain commands with `&&` only in bash/zsh. Windows PowerShell 5.1 (the Windows 10
> default) does not support `&&`, so run each command on its own line.

### 2. Upgrade pip (required)

Editable installs need **pip ≥ 21.3** (PEP 660). Older pip (e.g. 20.x) fails with
*"File setup.py not found … editable mode currently requires a setup.py based
build"*. Upgrade first:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install the package

**Windows (PowerShell or cmd.exe):**

```powershell
pip install -e .            # core (openai, pydantic, numpy, Pillow)
pip install -e ".[chem]"    # RDKit SMILES canonicalization (recommended)
pip install -e ".[vision]"  # OpenCV/scikit-image (optional local preprocessing)
pip install -e ".[dev]"     # pytest
pip install -e ".[all]"     # everything (used by the tests)
```

**macOS / Linux (bash/zsh):**

```bash
pip install -e .            # core (openai, pydantic, numpy, Pillow)
pip install -e '.[chem]'    # RDKit SMILES canonicalization (recommended)
pip install -e '.[vision]'  # OpenCV/scikit-image (optional local preprocessing)
pip install -e '.[dev]'     # pytest
pip install -e '.[all]'     # everything (used by the tests)
```

> The double quotes (`".[all]"`) matter in PowerShell because unquoted square
> brackets are treated as special characters.

### 4. Set your API key

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = "sk-..."          # current session only
setx OPENAI_API_KEY "sk-..."            # persist for future sessions (reopen shell)
```

**macOS / Linux (bash/zsh):**

```bash
export OPENAI_API_KEY="sk-..."
```

> Running as a Cursor Cloud Agent? Add `OPENAI_API_KEY` under
> **Cloud Agents → Secrets** in the Cursor dashboard so it is injected into the VM.

## Usage

Command line:

```bash
parse-notebook-page path/to/page.png -o result.json
# or:
python scripts/run_page.py path/to/page.png --model gpt-4o
```

Python API:

```python
from notebook_parser import NotebookPipeline, PipelineConfig

pipeline = NotebookPipeline(PipelineConfig(page_id="page_57"))
result = pipeline.run("path/to/page.png")
print(result.to_json())
```

Offline / custom engine (no network) — inject any object implementing
`LLMEngine.extract(...)`, e.g. the bundled `StubEngine` or your own model:

```python
from notebook_parser import NotebookPipeline, StubEngine
pipeline = NotebookPipeline(engine=StubEngine(canned_responses))
```

## Output schema

The result serializes to the JSON in §2 of the spec: `page_id`, `document_type`,
`layout`, `transcript`, `symbols`, `chemistry` (`reagents`, `formulas`,
`structures`, `concentrations`), `experiment` (`goal`, `conditions`, `procedure`,
`observations`, `results`), and `confidence` (`overall` + per-field). Drawn
molecules appear under `chemistry.structures` as SMILES + name with an `uncertain`
flag.

## Determinism, cost, and latency

- Each page makes **three** model calls (transcription, chemistry, experiment).
- `temperature=0` + a fixed `seed` give best-effort reproducibility; hosted models
  are not bit-exact, so determinism is not guaranteed.
- Image long side is capped (default 1600 px) to bound token cost; tune via
  `PipelineConfig.imaging.max_long_side`.

## Evaluation

```bash
python scripts/evaluate.py path/to/dataset/   # <name>.{png,jpg} + <name>.gt.json
```

Implements CER, WER, and reagent/formula/symbol F1 (§5, §13); the metric functions
are importable and unit-tested. The dataset driver needs an API key (it runs the
real pipeline).

## Testing

```bash
pytest
```

The suite runs **fully offline** via `StubEngine` (no API key or network needed):
image conditioning, the LLM seam, all three passes (response→canonical mapping and
SMILES canonicalization), validation, and a deterministic end-to-end run.

## Troubleshooting installation

**`The token '&&' is not a valid statement separator`** — Windows PowerShell 5.1
doesn't support `&&`. Run each command on its own line.

**`File "setup.py" not found … editable mode currently requires a setup.py based
build`** — pip older than 21.3. Run
`python -m pip install --upgrade pip setuptools wheel` first.

**`SSLError(SSLEOFError… EOF occurred in violation of protocol)` /
`connection broken` / `Could not fetch URL https://pypi.org/...`** — pip can't reach
PyPI; this is an environment/network issue. Common Windows causes/fixes:

- *Antivirus/firewall HTTPS scanning*: temporarily disable "HTTPS/SSL scanning",
  install, then re-enable.
- *Corporate/VPN proxy*: switch networks (e.g. phone hotspot) to confirm, or
  `pip install -e ".[all]" --proxy http://USER:PASS@HOST:PORT`.
- *Flaky connection*: `pip install -e ".[all]" --retries 10 --timeout 60`.
- *Intercepted certificate*: add
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

**`OPENAI_API_KEY is not set`** — export the key (see step 4) or add it as a Cloud
Agent secret.

## Limitations

- Accuracy depends on the chosen model and image quality; very faint or ambiguous
  handwriting may still be misread (alternatives and confidences are exposed, and
  low-confidence lines are flagged for review).
- SMILES for hand-drawn structures are best-effort and flagged `uncertain` when the
  drawing is ambiguous; enable the `chem` extra for RDKit validation.
- Hosted-model outputs are not bit-for-bit deterministic.
