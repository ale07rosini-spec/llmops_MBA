"""A4 – Publica o modelo champion no Hugging Face Hub (repo Git + Model Card + tag de versão).

Requer a variável HF_TOKEN (Settings > Access Tokens, permissão "write").
Uso:  python hub_publish.py --repo-id SEU_USUARIO/wine-quality-classifier
"""
import argparse
import os
import sys

import common

CARD = """---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
tags: [sklearn, tabular-classification, mlops]
metrics: [f1, accuracy]
---
# {name}

Classificador RandomForest (dataset `sklearn.load_wine`) usado na aula de Model Registry e LLMOps.

## Rastreabilidade (vem do MLflow Registry)
- Versão no registry: **v{version}** (alias `champion` no momento da publicação)
- F1 macro no teste: **{f1:.3f}** · acurácia: **{acc:.3f}**
- Hiperparâmetros: `n_estimators={n_estimators}`, `max_depth={max_depth}`
- Commit do código: `{commit}`

## Uso
```python
import joblib
model = joblib.load("model.joblib")
```

## Limitações
Modelo didático: dataset pequeno (178 amostras); não use em produção.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True)
    args = ap.parse_args()
    if not os.getenv("HF_TOKEN"):
        sys.exit("Defina HF_TOKEN (ou rode `hf auth login`) antes de publicar.")

    common.setup_mlflow("wine-registry")
    import joblib
    import mlflow.sklearn
    from huggingface_hub import HfApi
    from mlflow import MlflowClient

    client = MlflowClient()
    champ = client.get_model_version_by_alias(common.MODEL_NAME, common.CHAMPION)
    run = client.get_run(champ.run_id)
    model = mlflow.sklearn.load_model(f"models:/{common.MODEL_NAME}@{common.CHAMPION}")

    folder = os.path.join(common.ensure_out(), "hub")
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, "model.joblib"))
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD.format(
            name=common.MODEL_NAME, version=champ.version,
            f1=run.data.metrics["f1_macro"], acc=run.data.metrics["accuracy"],
            n_estimators=run.data.params["n_estimators"], max_depth=run.data.params["max_depth"],
            commit=run.data.tags.get("git_commit", "?"),
        ))

    api = HfApi()  # lê HF_TOKEN do ambiente
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)  # privado até revisar
    api.upload_folder(folder_path=folder, repo_id=args.repo_id, repo_type="model",
                      commit_message=f"{common.MODEL_NAME} v{champ.version} (champion)")
    api.create_tag(args.repo_id, tag=f"v{champ.version}", repo_type="model")
    print(f"Publicado: https://huggingface.co/{args.repo_id}  (tag v{champ.version})")


if __name__ == "__main__":
    main()
