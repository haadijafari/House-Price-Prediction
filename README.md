# House Price Prediction — Tehran

Predict house prices in Tehran using structured property features. Feature engineering, model training, evaluation and interpretation.

## Project overview

This repository contains a Jupyter Notebook that walks through a regression project to predict house prices in Tehran from property attributes (e.g., area, neighborhood, number of rooms, age). The goal is to deliver a reproducible model showing how raw data is transformed and used to train and evaluate predictive models suitable for deployment or further experimentation.

## Prerequisites

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [Git](https://git-scm.com/)

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/haadijafari/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Install dependencies

[uv](https://docs.astral.sh/uv/) handles the virtual environment and dependency resolution automatically:

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv`, and installs all project and dev dependencies.

### 3. Pull the dataset with DVC

The dataset is tracked with [DVC](https://dvc.org/) and stored on Google Drive. To download it:

```bash
uv run dvc pull
```

On the first run, a browser window will open for Google Drive authentication. After authenticating, the dataset (`data/housePrice.csv`) will be downloaded automatically.

### 4. Run the notebook

```bash
uv run jupyter notebook
```

Then open `House_Price.ipynb` in the browser to explore the full analysis pipeline.

## Repository contents

| File / Directory | Description |
|---|---|
| `House_Price.ipynb` | End-to-end analysis and modeling notebook (EDA → preprocessing → models → evaluation → interpretation) |
| `data/housePrice.csv` | Raw dataset (tracked by DVC, not stored in Git) |
| `data/housePrice.csv.dvc` | DVC metadata file pointing to the remote dataset |
| `.dvc/config` | DVC remote storage configuration (Google Drive) |
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Locked dependency versions for reproducibility |
| `CONTRIBUTING.md` | Commit message and branching guidelines |

## Data

- Source: [Kaggle](https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran)
- Dataset Features:
  - `Area`: House area (m²)
  - `Room`: Number of bedrooms
  - `Parking`: Does it have parking or not
  - `Warehouse`: Does it have warehouse or not
  - `Elevator`: Does it have elevator or not
  - `Address`: Approximate address in Tehran
  - `Price`: Price of the house in Toman
  - `Price(USD)`: Price of the house in US Dollar

- Target: `Price` or `Price(USD)`

## Skills & technologies

- **Language:** Python 3.13+
- **Package management:** uv
- **Data versioning:** DVC (Google Drive remote)
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Concepts:** Regression, feature engineering, EDA

## Contact

- Author: Haadi Jafari
- GitHub: [haadijafari](https://github.com/haadijafari)
- Email: [haadijafari2003@gmail.com](mailto:haadijafari2003@gmail.com)
