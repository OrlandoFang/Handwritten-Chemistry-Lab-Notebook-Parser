# Handwritten Chemistry Lab Notebook Parser

A multi-pass, multimodal pipeline that converts a scanned handwritten chemistry
lab notebook page into structured, machine-readable JSON. It combines classical
image processing, layout analysis, (pluggable) handwriting transcription,
scientific-symbol recovery, chemistry-aware extraction, and a constrained
experiment summarizer. See [`spec.md`](spec.md) for the full design.

## Why this design

The spec calls for trained ML components (a handwriting OCR model, a layout
detector, a drawing→SMILES extractor). Rather than hard-depend on weights that
may be unavailable, every model-backed stage is programmed against a small
**interface** and ships with a **deterministic, zero-dependency default backend**
plus optional real backends (Tesseract for OCR, RDKit for chemistry). The result:

- the whole pipeline runs end to end and emits the exact spec schema anywhere,
- heavy/optional dependencies are imported lazily and degrade gracefully,
- a trained model can be dropped into any stage without touching the rest,
- output is deterministic for a fixed config + input.

This honors the spec's modularity and determinism constraints (§11) while staying
honest about which parts need trained models to reach production accuracy.

## Architecture

The staged pipeline (§3, §7), each stage swappable:

```
load → preprocess → layout → transcribe → symbols → chemistry → semantics
     → confidence → validate
```

| Stage | Module | Default backend |
|-------|--------|-----------------|
| Image normalization | `preprocessing/` | OpenCV/scikit-image (deskew, denoise, illumination, binarize) |
| Layout + reading order | `layout/` | Connected-component heuristics (`HeuristicLayoutDetector`) |
| Handwriting OCR | `ocr/` | `Recognizer` protocol; Tesseract if installed, else review-flagged fallback |
| Symbol recovery | `symbols/` | Rule-based repair + unit/Greek/notation catalog |
| Chemistry extraction | `chemistry/` | Element-validated formula/reagent/concentration parsing; RDKit canonicalization; heuristic drawing detection |
| Experiment semantics | `semantics/` | Constrained, evidence-only cue classifier |
| Confidence + schema | `validation/` | Weighted per-field scoring + Pydantic schema enforcement |

All artifacts are typed Pydantic models (`types.py`) and carry provenance
(bounding boxes, candidate alternatives, applied corrections) so every output
traces back to an image region (§9).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (numpy, pydantic, networkx)
pip install -e '.[vision]'  # OpenCV + scikit-image preprocessing/layout
pip install -e '.[chem]'    # RDKit chemistry canonicalization
pip install -e '.[ocr]'     # pytesseract backend (also needs the tesseract binary)
pip install -e '.[dev]'     # pytest
# or everything used by the tests:
pip install -e '.[all]'
```

## Usage

Command line:

```bash
parse-notebook-page path/to/page.png -o result.json
# or, equivalently:
python scripts/run_page.py path/to/page.png --ocr-backend auto
```

Python API:

```python
from notebook_parser import NotebookPipeline, PipelineConfig

pipeline = NotebookPipeline(PipelineConfig(page_id="page_01"))
result = pipeline.run("path/to/page.png")
print(result.to_json())
```

Injecting a custom recognizer (any object with `recognize(crop) -> RecognitionOutput`):

```python
from notebook_parser.ocr import SequenceRecognizer  # or your trained model
result = pipeline.run(image, recognizer=SequenceRecognizer(known_lines))
```

## Output schema

The result serializes to the JSON in §2 of the spec: `page_id`, `document_type`,
`layout`, `transcript`, `symbols`, `chemistry` (`reagents`, `formulas`,
`structures`, `concentrations`), `experiment` (`goal`, `conditions`, `procedure`,
`observations`, `results`), and `confidence` (`overall` + per-field).

## Evaluation

```bash
python scripts/evaluate.py path/to/dataset/   # <name>.{png,jpg} + <name>.gt.json
```

Implements CER, WER, and reagent/formula/symbol F1 (§5, §13). The metric
functions are importable and unit-tested.

## Training (optional models)

`scripts/train_layout.py` and `scripts/train_ocr.py` define the training contract
for the model-backed stages. No weights or datasets are bundled, so they validate
prerequisites and document the required dataset/interface rather than fabricating
a run. Trained models load behind the `LayoutDetector` / `Recognizer` protocols.

## Testing

```bash
pytest
```

Covers preprocessing, layout/reading-order, OCR decode/correct, symbol recovery,
chemistry parsing/normalization, semantics, validation, and a deterministic
end-to-end integration run.

## Limitations

- The default OCR backend cannot read handwriting; install a real handwriting
  model (behind `Recognizer`) or Tesseract for printed text. Without one, the
  transcript is empty and flagged for human review (by design, not invented).
- Hand-drawn structure recovery emits a partial, explicitly uncertain record
  (region + rough estimates), not full SMILES, unless a structure model is added.
- Layout detection is heuristic; a trained detector improves messy pages.
