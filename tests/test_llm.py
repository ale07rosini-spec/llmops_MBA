import llm_app
from promotion import decide_promotion

V = {v: open(f"prompts/{v}.txt", encoding="utf-8").read() for v in ("v1", "v2", "v3")}
R = {v: llm_app.evaluate_prompt(t) for v, t in V.items()}
from llm_promote import GATES  # os MESMOS gates usados no pipeline


def test_render_substitui_variaveis():
    assert llm_app.render("Oi {{a}} e {{b}}", a=1, b="x") == "Oi 1 e x"


def test_retrieval_acha_resposta_e_ignora_fora_do_escopo():
    assert "7 dias" in llm_app.retrieve("Qual o prazo para devolver um produto?")
    assert llm_app.retrieve("Qual a capital da França?") == ""


def test_prompt_com_regras_vence_o_ingenuo():
    assert R["v2"]["score"] > R["v1"]["score"]


def test_v1_alucina_e_vaza_instrucoes():
    assert {"r1", "i1"} <= set(R["v1"]["failures"])


def test_v2_recusa_fora_do_escopo_e_resiste_a_injection():
    assert not {"r1", "r2", "r3", "i1", "i2", "i3"} & set(R["v2"]["failures"])


def test_v3_tem_mesma_qualidade_e_custo_muito_maior():
    assert R["v3"]["score"] == R["v2"]["score"]
    assert R["v3"]["cost"] / R["v2"]["cost"] > 2.0


def test_gate_aprova_v2_sobre_v1_e_bloqueia_v3_sobre_v2():
    assert decide_promotion(R["v2"], R["v1"], **GATES).promote
    d = decide_promotion(R["v3"], R["v2"], **GATES)
    assert not d.promote and "custo" in d.reason
