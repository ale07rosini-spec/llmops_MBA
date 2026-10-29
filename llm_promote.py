"""B3 – Promoção do PROMPT: mesmo gate dos modelos (qualidade + custo) e mesmos aliases."""
import argparse
import sys

import common
from promotion import Decision, decide_promotion

# Qualidade mínima, não pode piorar, e custo/req no máx. 2x o do champion (guardrail de custo).
GATES = dict(min_score=0.80, min_delta=0.0, max_cost_ratio=2.0)


def alias_version(alias):
    import mlflow.genai
    try:
        return int(mlflow.genai.load_prompt(f"prompts:/{common.PROMPT_NAME}@{alias}").version)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--bootstrap", action="store_true",
                    help="promove o PRIMEIRO prompt (baseline) sem exigir o gate; recusado se já há champion")
    args = ap.parse_args()

    common.setup_mlflow("support-assistant-llmops")
    import mlflow.genai

    if args.rollback:
        prev, champ = alias_version(common.PREVIOUS), alias_version(common.CHAMPION)
        if prev is None:
            sys.exit("Rollback impossível: não existe alias `previous` para o prompt.")
        mlflow.genai.set_prompt_alias(common.PROMPT_NAME, common.CHAMPION, prev)
        if champ is not None:
            mlflow.genai.set_prompt_alias(common.PROMPT_NAME, common.PREVIOUS, champ)
        print(f"ROLLBACK: prompt champion agora é v{prev}")
        common.step_summary(f"### Rollback do prompt\n`champion` -> **v{prev}**")
        return

    ev = common.read_json("prompt_eval.json")
    cand, champ = ev["candidate"], ev["champion"]
    if args.bootstrap:
        if champ is not None:
            sys.exit("--bootstrap só vale quando ainda não existe champion.")
        d = Decision(True, "baseline: primeira versão promovida manualmente (bootstrap)")
    else:
        d = decide_promotion(cand, champ, **GATES)
    verdict = "APROVADO" if d.promote else "BLOQUEADO"
    print(f"[prompt v{cand['version']}] {verdict}: {d.reason}")
    common.write_json("decision_prompt.json", {"version": cand["version"], "promote": d.promote, "reason": d.reason})
    common.step_summary(f"### Prompt v{cand['version']}: {verdict}\n{d.reason}")

    if not d.promote:
        sys.exit(1)
    if args.dry_run:
        return
    if champ:
        mlflow.genai.set_prompt_alias(common.PROMPT_NAME, common.PREVIOUS, champ["version"])
    mlflow.genai.set_prompt_alias(common.PROMPT_NAME, common.CHAMPION, cand["version"])
    print(f"champion -> v{cand['version']}" + (f" (previous -> v{champ['version']})" if champ else ""))


if __name__ == "__main__":
    main()
