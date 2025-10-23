import numpy as np

class RuleEngine:
    """
    Regla base leve/grave por umbrales físicos. El modelo ML complementa y corrige.
    Etiquetas objetivo: {0: correcto, 1: leve, 2: grave}
    """
    def __init__(self, conf_rules):
        self.r = conf_rules

    def classify_segment(self, feat_row):
        # ejemplos simples (ajustar por movimiento en fases posteriores):
        knee_depth = min(180 - feat_row.get("knee_left_p10",180),
                         180 - feat_row.get("knee_right_p10",180))
        elbow_straight = max(feat_row.get("elbow_left_p90",0),
                             feat_row.get("elbow_right_p90",0))
        trunk_tilt = abs(90 - feat_row.get("shoulder_p50", 90))
        jitter = max(feat_row.get("knee_left_max_acc",0),
                     feat_row.get("knee_right_max_acc",0))

        # grave si no se alcanza profundidad mínima o hay gran inestabilidad
        if knee_depth < self.r["knee_depth_min_deg"] or jitter > self.r["jitter_max_deg_s2"]*1.5:
            return 2
        # leve si brazo no llega a extensión o tronco inclinado
        if (180 - elbow_straight) > self.r["elbow_straight_max_deg"] or trunk_tilt > self.r["trunk_tilt_max_deg"]:
            return 1
        return 0

    @staticmethod
    def to_deduction(label):
        return {0: 0.0, 1: 0.1, 2: 0.3}[label]
