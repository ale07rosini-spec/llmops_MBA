"""A2 – Promoção do MODELO: aplica o gate e move o alias `champion` (ou faz rollback)."""
import argparse
import sys

import common
from promotion import decide_promotion

# Gates do modelo: o mínimo absoluto protege o primeiro deploy; o delta compara com o champion.
GATES = dict(min_score=0.90, min_delta=0.0)


def get_alias(client, alias):
    try:
        return client.get_model_version_by_alias(common.MODEL_NAME, alias)
    except Exception:  # alias ainda não existe
        return None


def champion_score(client, champ):
    """Métrica do champion = a que foi LOGADA no run que o gerou (mesmo dataset e split)."""
    return client.get_run(champ.run_id).data.metrics["f1_macro"]


def rollback(client):
    prev, champ = get_alias(client, common.PREVIOUS), get_alias(client, common.CHAMPION)
    if prev is None:
        sys.exit("Rollback impossível: não existe alias `previous`.")
    client.set_registered_model_alias(common.MODEL_NAME, common.CHAMPION, prev.version)
    if champ is not None:  # troca: permite 'roll-forward' se necessário
        client.set_registered_model_alias(common.MODEL_NAME, common.PREVIOUS, champ.version)
    print(f"ROLLBACK: champion agora é v{prev.version}")
    common.step_summary(f"### Rollback do modelo\n`champion` -> **v{prev.version}**")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="só decide, não move alias (usado na CI)")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()

    common.setup_mlflow("wine-registry")
    from mlflow import MlflowClient
    client = MlflowClient()

    if args.rollback:
        return rollback(client)

    cand = common.read_json("candidate.json")
    champ = get_alias(client, common.CHAMPION)
    champ_dict = {"score": champion_score(client, champ), "cost": None} if champ else None

    d = decide_promotion({"score": cand["score"], "cost": None}, champ_dict, **GATES)
    verdict = "APROVADO" if d.promote else "BLOQUEADO"
    print(f"[modelo v{cand['version']}] {verdict}: {d.reason}")
    common.write_json("decision_model.json", {"version": cand["version"], "promote": d.promote, "reason": d.reason})
    common.step_summary(f"### Modelo v{cand['version']}: {verdict}\n{d.reason}")

    if not d.promote:
        sys.exit(1)           # pipeline vermelho = candidato rejeitado
    if args.dry_run:
        return
    if champ is not None:
        client.set_registered_model_alias(common.MODEL_NAME, common.PREVIOUS, champ.version)
    client.set_registered_model_alias(common.MODEL_NAME, common.CHAMPION, cand["version"])
    print(f"champion -> v{cand['version']}" + (f" (previous -> v{champ.version})" if champ else ""))


if __name__ == "__main__":
    main()
