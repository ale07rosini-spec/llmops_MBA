import reg_train


def test_modelo_bom_passa_do_minimo_do_gate():
    _, m, _ = reg_train.train_model(100, 5)
    assert m["f1_macro"] >= 0.90


def test_modelo_fraco_fica_abaixo_do_gate():
    _, m, _ = reg_train.train_model(2, 1)
    assert m["f1_macro"] < 0.90


def test_treino_e_reprodutivel():
    _, a, _ = reg_train.train_model(10, 3)
    _, b, _ = reg_train.train_model(10, 3)
    assert a == b
