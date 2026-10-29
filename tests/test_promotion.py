from promotion import decide_promotion

GATES = dict(min_score=0.8, min_delta=0.0, max_cost_ratio=2.0)


def test_primeiro_deploy_passa_nos_gates_absolutos():
    assert decide_promotion({"score": 0.9, "cost": 1.0}, None, **GATES).promote


def test_bloqueia_abaixo_do_minimo():
    d = decide_promotion({"score": 0.5, "cost": 1.0}, None, **GATES)
    assert not d.promote and "mínimo" in d.reason


def test_bloqueia_se_nao_supera_champion():
    d = decide_promotion({"score": 0.85, "cost": 1.0}, {"score": 0.9, "cost": 1.0}, **GATES)
    assert not d.promote


def test_empate_de_score_passa_com_delta_zero():
    assert decide_promotion({"score": 0.9, "cost": 1.0}, {"score": 0.9, "cost": 1.0}, **GATES).promote


def test_bloqueia_custo_acima_do_limite_mesmo_com_qualidade_igual():
    d = decide_promotion({"score": 0.9, "cost": 3.0}, {"score": 0.9, "cost": 1.0}, **GATES)
    assert not d.promote and "champion" in d.reason


def test_orcamento_absoluto():
    d = decide_promotion({"score": 0.9, "cost": 5.0}, None, min_score=0.8, max_cost=2.0)
    assert not d.promote
