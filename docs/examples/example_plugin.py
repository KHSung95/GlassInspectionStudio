"""Example third-party algorithm. Copy into <repo>/plugins/ to auto-register it."""
import numpy as np

from gistudio.algorithms.base import InspectionAlgorithm, InspectionResult, ParamSpec, register
from gistudio.algorithms.classical import ops
from gistudio.algorithms.classical.algorithms import _spot_candidates


@register
class DarkPixelThreshold(InspectionAlgorithm):
    algorithm_id = "plugin.dark_pixel"
    display_name = "Plugin example: dark pixel threshold"
    family = "classical"
    score_semantics = "(median - pixel) / noise sigma"
    params_spec = [ParamSpec("threshold", "float", 8.0, 0.5, 100, 0.5)]

    def _run(self, image, meta):
        x = ops.f32(image)
        s = (float(np.median(x)) - x) / ops.noise_sigma(image)
        return InspectionResult(_spot_candidates(self, image, s, self.params["threshold"], "Black Spot", 1, 400, 500),
                                {"dark_snr": s})
