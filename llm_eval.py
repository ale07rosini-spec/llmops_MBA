"""B2 – Avalia o prompt candidato (e o champion atual) no MESMO golden set; loga tudo no MLflow."""
import common
import llm_app


def load_template(version=None, alias=None):
    """Retorna (versão, template) ou None se o alias ainda não existir."""
    import mlflow.genai
    uri = f"prompts:/{common.PROMPT_NAME}/{version}" if version else f"prompts:/{common.PROMPT_NAME}@{alias}"
    try:
        p = mlflow.genai.load_prompt(uri)
    except Exception:
        return None
    return int(p.version), p.template


def main():
    mlflow = common.setup_mlflow("support-assistant-llmops")
    cand_v = common.read_json("prompt_candidate.json")["version"]
    _, cand_tpl = load_template(version=cand_v)
    cand = {"version": cand_v, **llm_app.evaluate_prompt(cand_tpl)}

    champ = None
    loaded = load_template(alias=common.CHAMPION)
    if loaded:
        champ = {"version": loaded[0], **llm_app.evaluate_prompt(loaded[1])}

    with mlflow.start_run(run_name=f"eval-prompt-v{cand_v}"):
        mlflow.set_tags({"prompt_version": str(cand_v), "git_commit": common.git_commit(), "llm": "mock"})
        mlflow.log_metrics({"score": cand["score"], "cost_per_request": cand["cost"],
                            "avg_prompt_tokens": cand["avg_prompt_tokens"],
                            "avg_latency_ms": cand["avg_latency_ms"]})

    common.write_json("prompt_eval.json", {"candidate": cand, "champion": champ})
    print(f"candidato v{cand_v}: score={cand['score']:.3f} custo=US$ {cand['cost']:.5f}/req falhas={cand['failures']}")
    if champ:
        print(f"champion  v{champ['version']}: score={champ['score']:.3f} custo=US$ {champ['cost']:.5f}/req")


if __name__ == "__main__":
    main()
