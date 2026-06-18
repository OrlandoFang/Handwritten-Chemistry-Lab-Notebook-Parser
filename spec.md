# Spec: Handwritten Chemistry Lab Notebook Parser

## 1. Objective
Build a robust parser that converts a scanned handwritten chemistry lab notebook page into structured, machine-readable output. The system must handle:

- messy handwritten text,
- scientific notation and symbols,
- hand-drawn chemical structures,
- experiment-level semantics such as goal, conditions, procedure, and results.

The main design principle is **multi-pass, multimodal extraction**: do not rely on a single OCR or vision-language pass. Instead, combine layout analysis, handwriting recognition, symbol recovery, chemistry-aware parsing, and structured post-processing.

---

## 2. Output Schema
The parser should emit JSON with the following top-level structure:

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
- `layout`: ordered regions with bounding boxes and reading order.
- `transcript`: line-level transcription with source region, confidence, and optional alternative candidates.
- `symbols`: normalized scientific symbols and units (e.g., `°C`, `μL`, `λmax`).
- `chemistry.structures`: machine-readable representation of drawn molecules when detectable, preferably as SMILES or a graph format.
- `experiment`: distilled semantic summary with evidence links back to the transcript.

---

## 3. System Overview
The system should use a staged pipeline:

1. **Image normalization**
2. **Layout segmentation**
3. **Text line detection**
4. **Handwriting OCR / transcription**
5. **Symbol and unit recovery**
6. **Chemistry-specific extraction**
7. **Experiment-level semantic synthesis**
8. **Confidence scoring and validation**

Each stage should preserve provenance so downstream stages can trace every output back to an image region.

---

## 4. Algorithm Choices

### 4.1 Image preprocessing
Use classic image processing before any recognition model.

Recommended steps:
- deskew and dewarp the page,
- denoise while preserving strokes,
- estimate background shading and normalize illumination,
- binarize adaptively when helpful, but retain a grayscale copy for models,
- enhance faint pencil/pen strokes with contrast normalization.

Suggested methods:
- Hough-based or projection-based deskewing,
- local thresholding (Sauvola / adaptive thresholding) for auxiliary masks,
- morphological filtering to remove scan noise,
- optional page boundary detection.

### 4.2 Layout analysis
Use a document layout detector to split the page into:
- handwritten text blocks,
- tables,
- chemical drawings,
- annotations / marginal notes,
- headings and labels.

Preferred approach:
- a lightweight object detection or segmentation model trained on notebook-style pages,
- supplemented by heuristic line grouping when model confidence is low.

Output should include bounding boxes and an estimated reading order.

### 4.3 Handwriting transcription
Use a handwriting-oriented recognizer rather than general OCR.

Recommended strategy:
- line-level recognition over word-level recognition,
- beam search with language model rescoring,
- character-level confidence tracking,
- fallback to segment-level transcription for low-confidence regions.

Why line-level first:
- handwritten chemistry notes often have irregular spacing,
- symbols and subscripts are easier to interpret with broader context,
- line-level decoding reduces broken tokens.

### 4.4 Scientific symbol recovery
Standard OCR often damages symbols like `°`, `μ`, `λ`, `θ`, superscripts, subscripts, and reaction arrows.

Add a dedicated symbol normalization pass that:
- detects likely scientific tokens,
- repairs common OCR confusions such as `u` vs `μ`, `oC` vs `°C`, `x10^-3` vs `×10^-3`,
- recognizes units and concentration patterns,
- preserves formatting semantics such as superscript/subscript when meaningful.

Use rule-based repair first, then a learned correction model for ambiguous cases.

### 4.5 Chemistry extraction
Chemistry content should be handled as a separate modality.

For hand-drawn structures:
- detect drawing regions,
- vectorize strokes where possible,
- infer atoms, bonds, ring patterns, and arrow annotations,
- convert to a graph representation and, when feasible, SMILES-like output.

For formulas and reagents:
- parse chemical names, abbreviations, stoichiometry, and concentrations,
- normalize common lab shorthand,
- link reagents to nearby procedure text and tables.

If full structure recovery is unreliable, the system should still output a partial structured representation with explicit uncertainty.

### 4.6 Experiment-level understanding
Use a final semantic layer that converts extracted text and chemistry into an experiment summary.

This layer should identify:
- objective / hypothesis,
- starting materials,
- experimental conditions,
- actions performed,
- measurements taken,
- observations,
- final outcome.

This can be implemented with a retrieval-augmented or constrained LLM summarizer that is only allowed to use evidence from the extracted transcript and chemistry fields.

---

## 5. Tooling Choices

### Core libraries
- **Python** as the main implementation language.
- **OpenCV** for preprocessing and geometric correction.
- **PyTorch** for recognition models.
- **scikit-image** for image cleanup and morphology.
- **pydantic** for typed output validation.
- **networkx** for chemical graph representation.
- **rdkit** for chemistry normalization and validation when structures can be inferred.

### Optional model components
- A handwriting OCR model fine-tuned on notebook pages.
- A layout detector trained on scientific and handwritten documents.
- A text-correction language model specialized for chemistry lab language.
- A structure-extraction module for drawings.

### Evaluation tools
- character error rate (CER),
- word error rate (WER),
- symbol-level accuracy,
- reagent/formula F1,
- structure recovery accuracy,
- experiment-summary factual consistency.

---

## 6. Project Structure

```text
project/
  README.md
  spec.md
  pyproject.toml
  src/
    notebook_parser/
      __init__.py
      config.py
      pipeline.py
      types.py
      preprocessing/
        deskew.py
        denoise.py
        normalize.py
      layout/
        detect.py
        reading_order.py
      ocr/
        recognize.py
        decode.py
        correct.py
      symbols/
        normalize.py
        patterns.py
      chemistry/
        extract.py
        structures.py
        normalize.py
      semantics/
        summarize.py
        evidence.py
      validation/
        schema.py
        confidence.py
  models/
    layout/
    ocr/
    chemistry/
  tests/
    data/
    test_preprocessing.py
    test_layout.py
    test_ocr.py
    test_symbols.py
    test_chemistry.py
    test_semantics.py
  scripts/
    run_page.py
    evaluate.py
    train_layout.py
    train_ocr.py
```

---

## 7. Pipeline Details

### `pipeline.py`
Orchestrates the full flow:
1. load page image,
2. preprocess,
3. detect layout regions,
4. run transcription on text regions,
5. detect and parse chemical drawings,
6. normalize symbols and formulas,
7. produce experiment summary,
8. validate JSON output.

### `types.py`
Define all intermediate and final dataclasses / Pydantic models:
- bounding boxes,
- text lines,
- symbol tokens,
- reagent mentions,
- chemical structures,
- experiment evidence items.

### `validation/`
Enforce schema correctness and confidence thresholds. Low-confidence extractions should still be emitted, but marked clearly.

---

## 8. Matching and Post-processing Rules

The parser should apply chemistry-aware post-processing rules such as:
- merging split tokens around units and superscripts,
- correcting common handwritten abbreviations,
- associating reagent names with nearby quantities,
- linking arrows, reaction conditions, and product annotations,
- preserving table rows and column semantics.

Examples:
- `10 mL` should be recognized as a single quantity unit pair.
- `0.5 M HCl` should remain chemically typed, not generic text.
- `Δ` or `heat` should be normalized into reaction-condition metadata when used in context.

---

## 9. Confidence and Human Review
Every extracted field should carry a confidence score.

Rules:
- High-confidence fields can be used directly.
- Medium-confidence fields should expose alternatives.
- Low-confidence regions should be flagged for human review.

The system should also preserve provenance:
- original image crop,
- bounding box,
- recognition candidates,
- correction steps applied.

---

## 10. Testing Strategy

### Unit tests
- image normalization on synthetic noisy scans,
- bounding-box ordering,
- symbol normalization edge cases,
- formula parsing and reagent extraction,
- schema validation.

### Integration tests
- full-page parsing on notebook samples,
- regression tests for tricky symbols,
- chemistry drawing detection on annotated examples.

### Error analysis
Track failures by category:
- handwriting ambiguity,
- symbol confusion,
- table misreading,
- chemistry-structure misses,
- semantic hallucination.

Use these categories to prioritize model retraining and rule updates.

---

## 11. Design Constraints

- The system should be modular so each component can be swapped independently.
- The output must be deterministic for the same model version and input image.
- The semantic summary must not invent chemistry details that are absent from the page.
- Any inferred result should be distinguishable from directly observed text.

---

## 12. Implementation Milestones

1. Build preprocessing and layout detection.
2. Implement line-level handwriting transcription.
3. Add symbol recovery and chemistry token normalization.
4. Add chemical drawing extraction.
5. Build semantic experiment summarization.
6. Add confidence scoring, validation, and tests.
7. Tune on hard notebook examples.

---

## 13. Notes on Evaluation Alignment
The evaluation emphasizes four layers:

- **Text**: maximize transcription fidelity.
- **Special symbols**: preserve scientific notation and formatting.
- **Chemistry**: extract structures, reagents, and concentrations.
- **Experiment**: infer intent, conditions, and outcome without hallucination.

This spec is designed so each layer is handled by a dedicated stage, with the later stages consuming structured evidence from earlier stages rather than raw pixels alone.

