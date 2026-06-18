"""Layout segmentation stage (§4.2).

Splits a page into typed regions (text/heading/table/drawing/annotation) with
bounding boxes, detects text lines inside text regions, and assigns a reading
order. The detector is swappable: :class:`HeuristicLayoutDetector` is the default
zero-dependency backend, but any object implementing :class:`LayoutDetector` can
be injected (e.g. a trained model).
"""

from .detect import HeuristicLayoutDetector, LayoutDetector, detect_text_lines
from .reading_order import assign_reading_order

__all__ = [
    "LayoutDetector",
    "HeuristicLayoutDetector",
    "detect_text_lines",
    "assign_reading_order",
]
