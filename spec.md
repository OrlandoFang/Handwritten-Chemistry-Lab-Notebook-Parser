# Spec: Handwritten Chemistry Lab Notebook Parser (multimodal LLM edition)

## 1. Objective
Convert a scanned/photographed handwritten chemistry lab notebook page into
structured, machine-readable JSON. The page may contain:

- messy cursive/print handwriting,
- scientific notation, units, and special symbols (`°C`, `μL`, `λmax`, `×10⁻³`),
- hand-drawn chemical structures (e.g. crown ethers, LiTFSI),
- arithmetic/derivations (e.g. Faradaic charge → moles),
- tables of measurements,
- experiment-level semantics: goal, conditions, procedure, observations, results.

### Why this rewrite
The previous version relied on classical CV + heuristic OCR. On real handwriting
that approach produces nonsense: there is no bundled handwriting model, so the
transcript is empty/garbage and every downstream stage degrades.

**New design principle:** use a hosted **multimodal large language model** (the
OpenAI API) as the recognition and reasoning engine, driven by a **multi-pass,
schema-constrained** pipeline. The model reads the page image directly; classical
image processing is reduced to light pre-conditioning. Each pass returns data via
**Structured Outputs** (a strict JSON schema), so the model cannot return free
text that breaks the contract.

GPU note: inference runs on OpenAI's servers, so a local GPU is **not required**.
A GPU, if present, only accelerates optional local image preprocessing.

---

## 2. Output Schema
The parser emits JSON with this exact top-level structure:

```json
{
  "page_id": "...",
  "document_type": "chemistry_lab_notebook",
  "layout": [ ... ],
  "transcript": [ ... ],
  "symbols": [ ... ],
  "chemistry": {
    "reagents": [ ... ],
    "formulas": [ ... ],
    "structures": [ ... ],
    "concentrations": [ ... ]
  },
  "experiment": {
    "goal": "...",
    "conditions": [ ... ],
    "procedure": [ ... ],
    "observations": [ ... ],
    "results": [ ... ]
  },
  "confidence": {
    "overall": 0.0,
    "fields": { ... }
  }
}
```

### Field conventions
- `layout`: ordered regions with bounding boxes (pixel coords) and reading order.
- `transcript`: line-level transcription with source region, confidence, and
  optional alternative readings.
- `symbols`: normalized scientific symbols/units with the raw form they replaced.
- `chemistry.structures`: machine-readable structures, preferably **SMILES**, plus
  a human-readable name and an explicit `uncertain` flag.
- `experiment`: distilled summary, each item linked to transcript evidence and
  marked `inferred` when it is a conclusion rather than a direct quote.
- `confidence`: per-field scores plus a weighted overall.

---

## 3. System Overview
Staged, multi-pass pipeline:

1. **Image conditioning** — load, optional deskew, downscale to a token/cost
   budget, encode to a base64 data URL.
2. **Transcription pass (vision)** — model returns `layout`, `transcript`, and
   recovered `symbols` from the image.
3. **Chemistry pass (vision)** — model returns reagents, formulas, drawn
   structures (SMILES), and concentrations, grounded in both image and transcript.
4. **Experiment pass (text)** — model synthesizes goal/conditions/procedure/
   observations/results **using only the transcript + chemistry as evidence** (no
   pixels), to minimize hallucination.
5. **Assembly + validation** — merge passes into the canonical schema, compute
   confidence, and validate.

Each pass is independent and swappable. Every pass is constrained by a Pydantic
schema via OpenAI Structured Outputs.

---

## 4. Engine and Passes

### 4.1 LLM engine
- A thin `LLMEngine` interface with one operation: given a system prompt, a user
  prompt, an optional image, and a Pydantic response schema, return a validated
  instance of that schema.
- Default implementation `OpenAIEngine` uses `chat.completions.parse` (vision +
  Structured Outputs) with `temperature=0` and a fixed `seed` for best-effort
  determinism, plus bounded retries with exponential backoff.
- The engine is **injectable**, so tests run fully offline with a stub and a real
  fine-tuned/alternative model can be swapped in without touching the passes.

### 4.2 Image conditioning
- Load via OpenCV/Pillow.
- Optional deskew (projection-profile) when OpenCV is available.
- Downscale longest side to a configurable cap (default 1600 px) to bound cost.
- Encode as PNG/JPEG base64 data URL; pass the pixel dimensions to the model so
  bounding boxes are in the correct coordinate space.

### 4.3 Transcription pass
Prompt the model to:
- read **every** line verbatim, preserving symbols/sub/superscripts,
- group lines into typed regions (text/heading/table/drawing/annotation/label),
- assign a human reading order,
- give per-line confidence and up to N alternative readings for ambiguous lines,
- list recovered scientific symbols/units with the raw and normalized form.

### 4.4 Chemistry pass
Prompt the model to extract, grounded in the image + transcript:
- reagents (name, normalized name, role, quantity, concentration, evidence),
- formulas (raw + normalized, validity),
- hand-drawn structures as **SMILES** + name, with an `uncertain` flag when the
  drawing is ambiguous,
- concentrations (value, unit, species).

SMILES are optionally canonicalized/validated with RDKit when installed.

### 4.5 Experiment pass
Text-only (transcript + chemistry as the sole evidence). Extract goal, conditions,
procedure, observations, results. Each item must cite the transcript line ids it
derives from and set `inferred=true` if it is a synthesized conclusion. The model
is instructed never to introduce chemistry facts absent from the evidence.

### 4.6 Confidence and review
- Per-field confidence aggregates the model-reported confidences.
- Overall is a weighted mean emphasizing transcription and chemistry.
- Low-confidence transcript lines are flagged `needs_review`.

---

## 5. Tooling
- **Python** main language.
- **openai** (>=1.40) Python SDK — multimodal Structured Outputs.
- **pydantic** v2 — schemas + validation.
- **OpenCV / Pillow** — optional image conditioning.
- **RDKit** — optional SMILES canonicalization/validation.
- **networkx** — optional molecular-graph utilities.

Evaluation tools: CER/WER, symbol F1, reagent/formula F1, structure validity,
experiment factual-consistency spot checks.

---

## 6. Project Structure

```text
project/
  README.md
  spec.md
  pyproject.toml
  src/notebook_parser/
    __init__.py
    config.py            # pipeline + LLM configuration
    types.py             # canonical Pydantic output schema
    imaging.py           # load / deskew / resize / base64 encode
    pipeline.py          # multi-pass orchestration
    cli.py
    llm/
      __init__.py
      client.py          # LLMEngine protocol, OpenAIEngine, StubEngine
      prompts.py         # system/user prompt templates per pass
      schemas.py         # structured-output response models per pass
    passes/
      __init__.py
      transcription.py    # layout + transcript + symbols
      chemistry.py        # reagents/formulas/structures/concentrations
      experiment.py       # goal/conditions/procedure/observations/results
    validation/
      schema.py          # schema enforcement
      confidence.py      # confidence aggregation + review flags
  scripts/
    run_page.py
    evaluate.py
  tests/
    data/
    test_imaging.py
    test_llm.py
    test_passes.py
    test_validation.py
    test_pipeline.py
```

---

## 7. Pipeline Details

### `pipeline.py`
1. condition image (load, deskew?, resize, encode),
2. transcription pass → layout + transcript + symbols,
3. chemistry pass → chemistry section,
4. experiment pass → experiment section,
5. assemble canonical result,
6. compute confidence + review flags,
7. validate schema and return.

### `types.py`
Canonical Pydantic models for boxes, regions, transcript lines, symbol tokens,
reagents, formulas, structures, concentrations, evidence items, confidence, and
the top-level result that serializes to §2.

### `llm/`
`schemas.py` defines per-pass response models that are Structured-Output friendly
(all fields required, optionals expressed as nullable). The passes map those to
the canonical `types.py` models.

---

## 8. Post-processing Rules
- Keep quantity/unit pairs intact (`10 mL`, `0.5 M HCl`).
- Preserve scientific notation and symbols verbatim in the transcript.
- Associate reagents with nearby quantities/concentrations (the model is prompted
  to do this; validated structurally).
- Normalize obvious shorthand (`v/v`, `rt`, `Δ`/heat → reaction condition).
- Canonicalize SMILES via RDKit when available; flag invalid SMILES as uncertain.

---

## 9. Confidence and Human Review
- Every field carries a confidence score.
- High → usable directly; medium → expose alternatives; low → flag for review.
- Provenance is preserved: bounding boxes, alternative readings, evidence links,
  and the pass that produced each datum.

---

## 10. Determinism and Safety
- `temperature=0` + fixed `seed` for best-effort reproducibility (hosted models
  are not bit-exact; this is acknowledged, not guaranteed).
- The experiment pass is evidence-constrained and receives no pixels, reducing
  hallucination; inferred items are explicitly marked.
- The API key is read from `OPENAI_API_KEY`; it is never logged or persisted.

---

## 11. Testing Strategy
- **Unit**: image conditioning, prompt/schema construction, response→canonical
  mapping, confidence aggregation, schema validation — all offline via a
  `StubEngine` (no network, deterministic).
- **Integration**: full pipeline with the stub engine asserting a schema-valid
  result and correct field wiring.
- **Live (manual/opt-in)**: a script run against the real API with a key, used for
  qualitative checks and the evaluation harness.

---

## 12. Milestones
1. Canonical schema + config + imaging.
2. LLM engine (OpenAI + stub) and prompts.
3. Transcription, chemistry, experiment passes.
4. Pipeline assembly + validation + confidence.
5. CLI + evaluation harness.
6. Tests (offline) green; live smoke test with a key.

---

## 13. Evaluation Alignment
Four layers, each owned by a pass:
- **Text** — transcription fidelity (CER/WER).
- **Symbols** — scientific-notation/unit preservation (symbol F1).
- **Chemistry** — reagents, concentrations, and SMILES structures (F1/validity).
- **Experiment** — intent/conditions/outcome without hallucination (factual
  consistency against cited evidence).
