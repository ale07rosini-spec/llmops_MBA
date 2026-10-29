"""Regra de promoção (o "gate"): a MESMA lógica serve para modelos e para prompts.

Candidato e champion são dicts {"score": float, "cost": float | None}.
 - gates absolutos  : min_score, max_cost        (valem mesmo sem champion)
 - gates relativos  : min_delta, max_cost_ratio  (comparam com o champion atual)
"""
from dataclasses import dataclass


@dataclass
class Decision:
    promote: bool
    reason: str


def decide_promotion(candidate, champion=None, *, min_score, min_delta=0.0,
                     max_cost=None, max_cost_ratio=None) -> Decision:
    score = candidate["score"]
    if score < min_score:
        return Decision(False, f"score {score:.3f} abaixo do mínimo {min_score:.3f}")

    cost = candidate.get("cost")
    if max_cost is not None and cost is not None and cost > max_cost:
        return Decision(False, f"custo {cost:.4f} acima do orçamento {max_cost:.4f}")

    if champion is None:
        return Decision(True, "sem champion atual: passou nos gates absolutos")

    if score < champion["score"] + min_delta:
        return Decision(False, f"score {score:.3f} não supera o champion "
                               f"({champion['score']:.3f} + delta {min_delta})")

    champ_cost = champion.get("cost")
    if max_cost_ratio is not None and cost is not None and champ_cost:
        ratio = cost / champ_cost
        if ratio > max_cost_ratio:
            return Decision(False, f"custo {ratio:.2f}x o do champion "
                                   f"(limite {max_cost_ratio:.2f}x)")

    return Decision(True, f"score {score:.3f} >= champion {champion['score']:.3f} e custo dentro do limite")
