"""A3 – Consome o modelo SEMPRE pelo alias (`models:/<nome>@champion`), nunca por caminho/versão fixa."""
import argparse
import sys

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alias", default=common.CHAMPION)
    ap.add_argument("--smoke", action="store_true", help="falha (exit 1) se a acurácia for baixa")
    args = ap.parse_args()

    common.setup_mlflow("wine-registry")
    import mlflow.pyfunc
    uri = f"models:/{common.MODEL_NAME}@{args.alias}"
    model = mlflow.pyfunc.load_model(uri)

    _, X_test, _, y_test = common.load_wine_split()
    pred = model.predict(X_test)
    acc = float((pred == y_test.to_numpy()).mean())
    print(f"{uri}: {len(pred)} previsões; primeiras = {[int(x) for x in pred[:5]]}; acurácia = {acc:.3f}")
    if args.smoke and acc < 0.90:
        sys.exit(f"Smoke test falhou: acurácia {acc:.3f} < 0.90")


if __name__ == "__main__":
    main()
