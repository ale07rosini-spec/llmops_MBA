"""Configuração e utilitários compartilhados por toda a prática (Registry + LLMOps)."""
import json
import os
import re
import subprocess

# --- Nomes no registry (mesma convenção para modelos e prompts) -------------
# STUDENT (opcional) prefixa os nomes: num servidor MLflow compartilhado pela turma,
# cada aluno tem o seu próprio modelo/prompt e os aliases `champion` não colidem.
_STUDENT = re.sub(r"[^a-z0-9-]+", "-", (os.getenv("STUDENT") or "").lower()).strip("-")
_PREFIX = f"{_STUDENT}-" if _STUDENT else ""

MODEL_NAME = f"{_PREFIX}wine-quality-classifier"
PROMPT_NAME = f"{_PREFIX}support-assistant"
CHAMPION = "champion"      # alias do que está em produção
PREVIOUS = "previous"      # alias para rollback rápido

# --- Backend do MLflow ------------------------------------------------------
# O Model Registry exige backend com banco de dados (SQLite serve para a aula).
# Para usar um servidor (local `mlflow server` ou DagsHub), defina MLFLOW_TRACKING_URI.
# Autenticação: o cliente MLflow lê MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD sozinho.
# `or` (e não só o default do getenv): num Secret ausente a variável chega como string vazia.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"

OUT_DIR = "out"


def setup_mlflow(experiment: str):
    import mlflow  # import tardio: os testes unitários não precisam do MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"{_PREFIX}{experiment}")
    return mlflow


def git_commit() -> str:
    """SHA do commit (no GitHub Actions vem de GITHUB_SHA; local, de `git rev-parse`)."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:8]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "local"


def load_wine_split():
    """Mesmo dataset e mesmo split em todos os scripts -> resultados comparáveis."""
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    X, y = load_wine(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def write_json(name: str, data: dict) -> str:
    path = os.path.join(ensure_out(), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def read_json(name: str) -> dict:
    with open(os.path.join(OUT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def step_summary(markdown: str):
    """Escreve no resumo do job do GitHub Actions (no-op fora do CI)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
