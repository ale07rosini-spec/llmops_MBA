"""B1 – Registra um prompt (arquivo do repositório) no MLflow Prompt Registry como nova versão."""
import argparse

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default="prompts/candidate.txt")
    ap.add_argument("--message", default=None, help="commit message da versão do prompt")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        template = f.read()

    common.setup_mlflow("support-assistant-llmops")
    import mlflow.genai

    prompt = mlflow.genai.register_prompt(
        name=common.PROMPT_NAME,
        template=template,                                   # variáveis no formato {{context}}, {{question}}
        commit_message=args.message or f"{args.file} @ {common.git_commit()}",
        tags={"git_commit": common.git_commit(), "source_file": args.file},
    )
    common.write_json("prompt_candidate.json", {"name": common.PROMPT_NAME, "version": int(prompt.version)})
    print(f"Prompt '{common.PROMPT_NAME}' registrado: v{prompt.version}")


if __name__ == "__main__":
    main()
