# Lab – Model Registry + LLMOps + CI/CD com GitHub Actions

Prática da aula **MLOps → Model Registry → LLMOps**. Modelos **e** prompts seguem o mesmo ciclo:
`registrar versão → avaliar → gate → alias champion → consumir pelo alias → rollback`.

| Bloco | Arquivos | O que faz |
|---|---|---|
| A – Model Registry | `reg_train.py` `reg_promote.py` `reg_serve.py` `hub_publish.py` | treina, registra, promove por alias, consome, publica no HF Hub |
| B – LLMOps | `llm_register.py` `llm_eval.py` `llm_promote.py` `llm_serve.py` `llm_app.py` | prompts versionados, golden set, guardrails, custo |
| C – CI/CD | `.github/workflows/model-cicd.yml` | testes → avaliação → gate → aprovação humana → promoção/rollback |
| Comum | `common.py` `promotion.py` `params.json` `prompts/*.txt` `data/*.json` `tests/` | gate compartilhado, dados, configuração versionada |

## Rodando localmente
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # testes não precisam de MLflow
```
**Bloco A – modelos** (3 execuções: baseline, melhora, regressão bloqueada)
```bash
python reg_train.py --n-estimators 5   --max-depth 1 && python reg_promote.py   # 1º deploy (F1≈0.92)
python reg_train.py --n-estimators 100 --max-depth 5 && python reg_promote.py   # melhora (F1=1.00) -> champion v2
python reg_train.py --n-estimators 2   --max-depth 1 && python reg_promote.py   # F1≈0.54 -> BLOQUEADO (exit 1)
python reg_serve.py                                    # consome models:/wine-quality-classifier@champion
python reg_promote.py --rollback                       # champion volta para `previous`
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
```
**Bloco B – prompts**
```bash
python llm_register.py --file prompts/v1.txt && python llm_eval.py && python llm_promote.py --bootstrap  # baseline
python llm_register.py --file prompts/v2.txt && python llm_eval.py && python llm_promote.py              # melhora -> champion
python llm_register.py --file prompts/v3.txt && python llm_eval.py && python llm_promote.py              # BLOQUEADO (custo 2.85x)
python llm_serve.py "Qual o prazo para devolver um produto?"
python llm_promote.py --rollback
```
O LLM é um **MockLLM determinístico** (sem API key; resultados idênticos para todos os alunos e no CI).
Para usar um LLM real: `pip install anthropic` e `LLM_PROVIDER=anthropic LLM_MODEL=<modelo> ANTHROPIC_API_KEY=...`
(caminho opcional, fora dos testes). O custo e a latência do mock são **simulados** (preços fictícios).

## Servidor MLflow (opcional, recomendado)
Por padrão cada script abre o `mlflow.db` (SQLite) **direto**. Com um servidor, os scripts viram *clientes* que falam
com uma API REST, que é a arquitetura real (cliente → tracking server → banco + artifact store).
Nenhum script muda: basta definir `MLFLOW_TRACKING_URI` (lido em `common.py`).

**Opção 1: servidor local** (bom para os Blocos A e B)
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000   # terminal 1 (deixe rodando)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000                                     # terminal 2  (Windows: set ...)
python reg_train.py && python reg_promote.py                                         # agora registra via HTTP
```
Os artifacts (modelos) passam pelo servidor (`mlflow-artifacts:/`) e vão para `mlartifacts/`.
**Servidor compartilhado pela turma?** Cada aluno define `export STUDENT=seu-nome`: os nomes viram
`seu-nome-wine-quality-classifier` / `seu-nome-support-assistant` e os aliases `champion` não colidem.

**Opção 2: DagsHub** (MLflow hospedado e gratuito, acessível pelo GitHub Actions, recomendado para o Bloco C)
O `localhost` não é alcançável pelo CI. Com o DagsHub, os jobs do pipeline e você enxergam **o mesmo** registry.

### Criando a conta e o servidor no DagsHub
1. Acesse <https://dagshub.com> → **Sign up** (o mais simples é *Sign up with GitHub*).
2. Clique em **Create** (ou **+**) → **New Repository**. Duas opções:
   - *Connect a repository* → GitHub → escolha o repositório deste lab (o DagsHub espelha o código); ou
   - crie um repositório vazio (ex.: `lab-mlops`). O código pode continuar só no GitHub: aqui usamos apenas o MLflow.
3. No repositório do DagsHub: botão **Remote** → aba **Experiments**. Copie a **MLflow Tracking remote**:
   `https://dagshub.com/<usuario>/<repo>.mlflow`
4. Gere o token: avatar → **Your settings** → **Tokens** (<https://dagshub.com/user/settings/tokens>).
   Já existe um *default access token*; copie-o. **O token é senha: não faça commit dele.**
5. Teste localmente:
   ```bash
   export MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/<repo>.mlflow
   export MLFLOW_TRACKING_USERNAME=<usuario>
   export MLFLOW_TRACKING_PASSWORD=<token>
   python reg_train.py --n-estimators 5 --max-depth 1 && python reg_promote.py
   python llm_register.py --file prompts/v1.txt && python llm_eval.py && python llm_promote.py --bootstrap
   ```
6. Abra a UI do MLflow: `https://dagshub.com/<usuario>/<repo>.mlflow` (ou *Experiments* → *Go to MLflow UI*).
   Em **Models** e **Prompts** você deve ver a versão 1 com o alias `champion`.
7. Para o pipeline: no GitHub, *Settings → Secrets and variables → Actions → New repository secret*:
   `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` (os mesmos valores do passo 5).

Com os secrets definidos, o workflow pula os passos de cache/artifact do SQLite (o estado fica no servidor) e o
rollback funciona a qualquer momento, sem depender do cache de 7 dias.
> PRs vindos de *fork* não recebem secrets: nesses casos o pipeline volta sozinho para o modo SQLite.

## Pipeline (Bloco C)
```
pull_request ─► test ─► evaluate (treina+registra, avalia, gates em --dry-run)         [CI: PR fica vermelho se reprovar]
push main    ─► test ─► evaluate ─► promote (Environment "production": aprovação humana) [CD: move alias champion + smoke test]
Run workflow (action=rollback) ───► rollback (aprovação humana): champion ← previous
```
Configuração no GitHub:
1. Suba esta pasta como um repositório (o diretório `.github/workflows/` precisa estar na raiz do repo; é uma pasta oculta).
2. *Settings → Environments → New environment* `production` → *Required reviewers* (você/colega). É isso que cria a pausa de aprovação.
3. (Opcional) *Settings → Secrets and variables → Actions*: secrets `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (servidor MLflow, ver [DagsHub](#criando-a-conta-e-o-servidor-no-dagshub)), secret `HF_TOKEN` e variable `HF_REPO_ID` (publica o champion no Hub).
4. Experimentos: edite `params.json` (modelo) ou `prompts/candidate.txt` (prompt) num branch e abra um PR. Troque por `prompts/v3.txt` e veja o gate de custo bloquear.

Sem `MLFLOW_TRACKING_URI`, o pipeline usa SQLite e repassa o estado entre jobs por *artifact* e entre execuções por *cache*
(didático: o cache expira em ~7 dias sem uso). Em projeto real, use um servidor MLflow remoto (ex.: DagsHub, acima).

## Convenções
- Aliases: `champion` (produção) e `previous` (rollback). Consumo sempre por `models:/<nome>@champion` e `prompts:/<nome>@champion`.
- Nomes: `wine-quality-classifier` e `support-assistant`; com `STUDENT=fulano`, viram `fulano-wine-quality-classifier` etc. (também nos experimentos).
- Gates: modelo `min_score=0.90`; prompt `min_score=0.80`, `max_cost_ratio=2.0`; ambos `min_delta=0` (não pode piorar). Ajuste em `reg_promote.py` / `llm_promote.py`.
- Um gate reprovado termina com **exit 1** (pipeline vermelho = candidato rejeitado).
