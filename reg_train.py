"""A1 – Treina o modelo, loga run + métricas e REGISTRA uma nova versão no Model Registry."""
import argparse
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import common


def train_model(n_estimators: int, max_depth: int):
    """Treino puro (sem MLflow) -> fácil de testar. Retorna (modelo, métricas, X_train)."""
    X_train, X_test, y_train, y_test = common.load_wine_split()
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    ).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, pred)),
    }
    return model, metrics, X_train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="params.json", help="hiperparâmetros versionados no Git")
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    args = ap.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth

    model, metrics, X_train = train_model(**params)

    mlflow = common.setup_mlflow("wine-registry")
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=f"rf-{params['n_estimators']}-{params['max_depth']}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": common.git_commit(),
            "dataset": "sklearn.load_wine",
            "split": "80/20 seed=42",
        })
        info = mlflow.sklearn.log_model(
            model,
            name="model",                              # MLflow 3: `name` (artifact_path foi descontinuado)
            registered_model_name=common.MODEL_NAME,   # cria/atualiza o Registered Model
            signature=infer_signature(X_train, model.predict(X_train)),
            # MLflow recente serializa com skops (mais seguro que pickle) e exige liberar os tipos da árvore.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:  # fallback: procura a versão criada por este run
        version = max(int(v.version) for v in client.search_model_versions(f"name='{common.MODEL_NAME}'")
                      if v.run_id == run.info.run_id)
    version = int(version)
    client.set_model_version_tag(common.MODEL_NAME, str(version), "git_commit", common.git_commit())

    cand = {"name": common.MODEL_NAME, "version": version, "run_id": run.info.run_id,
            "score": metrics["f1_macro"], "accuracy": metrics["accuracy"], "params": params}
    common.write_json("candidate.json", cand)
    print(f"Registrado {common.MODEL_NAME} v{version}  f1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
