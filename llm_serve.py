"""B4 – Usa o prompt SEMPRE pelo alias (`prompts:/<nome>@champion`) -> trocar de versão não exige deploy de código."""
import argparse

import common
import llm_app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", default="Qual o prazo para devolver um produto?")
    ap.add_argument("--alias", default=common.CHAMPION)
    args = ap.parse_args()

    common.setup_mlflow("support-assistant-llmops")
    import mlflow.genai
    prompt = mlflow.genai.load_prompt(f"prompts:/{common.PROMPT_NAME}@{args.alias}")

    r = llm_app.answer(prompt.template, args.question)
    print(f"[prompt v{prompt.version} @{args.alias}] {r['text']}")


if __name__ == "__main__":
    main()
