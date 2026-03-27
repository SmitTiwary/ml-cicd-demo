# 🌸 ML CI/CD Demo Project
### A teaching project for understanding CI/CD pipelines and Git branching in ML

---

## 📁 Project Structure

```
ml-cicd-demo/
├── src/
│   ├── model.py          # Iris classifier: train, evaluate, predict
│   ├── data_utils.py     # Data loading, validation, splitting
│   └── config.py         # Environment-specific config (dev/stg/prod)
│
├── tests/
│   ├── test_model.py         # Unit tests for model functions
│   ├── test_data_utils.py    # Unit tests for data utilities
│   └── test_integration.py   # Integration tests tagged by environment
│
├── models/               # Saved model artifacts (auto-created)
│
├── .github/workflows/
│   ├── ci-dev.yml        # CI for dev branch  (smoke tests)
│   ├── ci-stg.yml        # CI/CD for stg branch (integration + deploy)
│   └── cd-prod.yml       # CD for prod branch  (strict gates + release)
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🌿 Branch Strategy

```
feature/xyz  →  dev  →  stg  →  prod
```

| Branch | Purpose | Who pushes | CI/CD behaviour |
|--------|---------|-----------|-----------------|
| `dev`  | Active development | Developers freely | Fast smoke tests, lenient accuracy ≥ 0.80 |
| `stg`  | Pre-release / QA   | PR from dev only  | Full integration tests, accuracy ≥ 0.90, deploy to staging |
| `prod` | Live / production  | PR from stg only  | Strictest gates, accuracy ≥ 0.93, auto-tag release |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model locally
python src/model.py

# 3. Run ALL tests
pytest tests/ -v

# 4. Run tests for a specific environment
pytest tests/test_integration.py -v -m dev
pytest tests/test_integration.py -v -m stg
pytest tests/test_integration.py -v -m prod

# 5. Run with coverage report
pytest tests/ --cov=src --cov-report=term
```

---

## 🎓 Teaching Demo: Step-by-Step

### Step 1 — Work on the dev branch

```bash
git checkout dev

# Make a change, e.g. tweak the model hyperparameter in model.py
# Change: max_iter=100  →  max_iter=200

git add src/model.py
git commit -m "feat: increase max_iter to 200"
git push origin dev
```

👀 **Watch:** GitHub Actions fires `ci-dev.yml` → lint → unit tests → dev integration tests.

---

### Step 2 — Promote dev → stg

```bash
# On GitHub: open a Pull Request from dev → stg
# On merge:
git checkout stg
git merge dev
git push origin stg
```

👀 **Watch:** `ci-stg.yml` fires:
1. Lint + format check
2. Unit tests
3. Integration tests with **tighter** threshold (≥ 0.90)
4. Deploys model artifact to staging environment

---

### Step 3 — Promote stg → prod

```bash
# On GitHub: open a Pull Request from stg → prod
# Requires approval from a reviewer
# On merge:
git checkout prod
git merge stg
git push origin prod
```

👀 **Watch:** `cd-prod.yml` fires:
1. Full test suite (all environments)
2. Production quality gate (accuracy ≥ 0.93)
3. Security vulnerability scan
4. Deploys to production
5. Creates a release tag automatically

---

## 🧪 Understanding the Test Layers

### Unit Tests (`test_model.py`, `test_data_utils.py`)
- Test individual functions in isolation
- Fast — run in seconds
- Examples: does `validate_features` reject bad input? Does `preprocess` produce zero mean?

### Integration Tests (`test_integration.py`)
- Test the **full pipeline** end-to-end
- Tagged with `@pytest.mark.dev`, `@pytest.mark.stg`, `@pytest.mark.prod`
- Each environment has **different accuracy thresholds**:

```python
# DEV  — experimenting allowed
assert acc >= 0.80

# STG  — almost ready to ship
assert acc >= 0.90

# PROD — ship it!
assert acc >= 0.93
```

---

## ⚙️ Environment Config

The `ENV` environment variable controls which config is loaded:

```bash
ENV=dev  python src/model.py   # lenient thresholds, debug logging
ENV=stg  python src/model.py   # tighter thresholds, info logging
ENV=prod python src/model.py   # strict thresholds, warning-only logging
```

In CI/CD, the `env:` block in each workflow file sets this automatically.

---

## 💡 Key Concepts to Highlight to Students

1. **Same code, different thresholds** — `config.py` shows how one codebase adapts to environments.
2. **Tests as quality gates** — CI literally blocks a merge if accuracy drops.
3. **Progressive strictness** — dev is permissive so devs can experiment; prod is strict so users aren't affected.
4. **Artifacts** — trained model files (`model.pkl`, `scaler.pkl`) are built in CI and uploaded, not committed to Git.
5. **Release tagging** — prod CD auto-tags every successful deploy for traceability.
