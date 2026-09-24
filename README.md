# 🏡 Tehran Real Estate Valuation & Inflation-Hedging Engine

Predict residential property prices across Tehran, Iran using machine learning. Features intrinsic USD modeling to insulate valuations from Rial currency hyperinflation, dynamic exchange-rate adaptation via environment variables, zero-leakage scikit-learn pipelines, and an interactive Streamlit web application.

---

## 🌟 Key Highlights

- **Inflation-Resistant Modeling**: By training directly on `Price(USD)`, the system captures fundamental real estate purchasing power, bypassing nominal Iranian Rial/Toman inflation drift.
- **Dynamic Real-Time Conversion**: Uses `USD_TO_TOMAN_RATE` (configurable via `.env` / environment variable) to project real-time Toman market prices for any macroeconomic timeframe.
- **Champion Model Performance**: **Random Forest Regressor** achieves a holdout test **$R^2$ of 0.822** across the full Tehran residential market (3,450 authentic listings).
- **Zero Data Leakage**: Custom feature engineering and `ColumnTransformer` with `OneHotEncoder(min_frequency=5)` and `StandardScaler` fitted strictly on training folds.
- **Interactive Streamlit Web App**: Launch an interactive real estate valuation dashboard with live exchange rate controls via `uv run streamlit run app.py`.
- **100% Self-Contained Notebook**: [`House_Price.ipynb`](House_Price.ipynb) contains the complete analysis from raw data ingestion to model serialization with zero external script dependencies.

---

## 📊 Model Benchmark Results

Models were systematically evaluated under **5-Fold Cross-Validation** on training folds and tested on an independent **20% holdout test set**:

| Model Architecture | Holdout $R^2$ | MAE ($ USD) | RMSE ($ USD) | MedAE ($ USD) | MAPE (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Log Target)** 🏆 | **0.822** | **$51,889** | **$122,216** | **$18,851** | **35.1%** |
| **Linear Regression (Log Target)** | 0.748 | $53,129 | $145,424 | $16,318 | 29.8% |
| **Ridge Regression CV (Log Target)** | 0.740 | $53,834 | $147,966 | $16,761 | 30.0% |
| **ElasticNet CV (Log Target)** | 0.719 | $55,870 | $153,585 | $17,391 | 31.2% |
| **Baseline (Median Predictor)** | -0.110 | $141,496 | $305,522 | $59,800 | 119.5% |

---

## 📁 Repository Structure

```text
├── House_Price.ipynb          # 100% self-contained end-to-end data science notebook
├── app.py                     # Interactive Streamlit valuation web application
├── model.joblib               # Serialized champion model pipeline (trained on USD)
├── pyproject.toml             # Project dependencies and configuration (managed by uv)
├── uv.lock                    # Locked dependency versions for exact reproducibility
├── .env.example               # Template for environment variables (USD_TO_TOMAN_RATE)
├── .env                       # Local environment configuration
├── data/
│   ├── housePrice.csv         # Raw dataset (tracked by DVC)
│   └── housePrice.csv.dvc     # DVC metadata pointer
└── README.md                  # Project overview and instructions
```

---

## 🚀 Getting Started

### 1. Prerequisites
- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager
- [Git](https://git-scm.com/)

### 2. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/haadijafari/House-Price-Prediction.git
cd House-Price-Prediction
uv sync
```

### 3. Pull Dataset with DVC
```bash
uv run dvc pull
```

### 4. Configure Currency Conversion Rate
Copy the example environment file:
```bash
cp .env.example .env
```
Adjust `USD_TO_TOMAN_RATE` to your desired market exchange rate (e.g. `60000` Tomans/USD).

### 5. Launch the Streamlit Web Application
Run the interactive property valuation dashboard:
```bash
uv run streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom property specs, compare against neighborhood median prices, and adjust exchange rates in real time.

### 6. Explore the Jupyter Notebook
Explore the full narrative, 10 before-and-after visual diagnostics, and training pipelines:
```bash
uv run jupyter notebook
```
Open [`House_Price.ipynb`](House_Price.ipynb) to inspect all pre-rendered charts and markdown documentation.

---

## 💡 Economic Methodology & Feature Engineering

### The Inflation Hedging Rationale
In high-inflation economies like Iran, real estate prices denominated in Iranian Tomans/Rials experience continuous nominal inflation. Training models directly in local currency causes rapid parameter decay. By modeling **`Price(USD)`** with logarithmic scaling ($\log(1 + \text{Price}_{USD})$):
1. The target variable is normalized from a heavy right-skew of **4.77** to a near-Gaussian **0.06**.
2. Intrinsic property value is preserved across years.
3. Real-time Toman market prices are calculated via:
   $$\text{Price}_{\text{Toman}} = \widehat{\text{Price}}_{\text{USD}} \times \text{USD\_TO\_TOMAN\_RATE}$$

### Engineered Features
- **`Area_per_Room`**: Living space density per bedroom ($\text{Area} / \max(\text{Room}, 1)$), separating compact units from luxury layouts.
- **`Total_Amenities`**: Composite score summing $\text{Parking} + \text{Warehouse} + \text{Elevator}$ ($0$ to $3$).
- **Categorical Frequency Thresholding**: Scikit-Learn's `OneHotEncoder(min_frequency=5)` groups rare neighborhoods into an infrequent category, preventing high-cardinality overfitting.

---

## 🛠️ Skills & Technologies
- **Language & Runtime:** Python 3.13+, uv
- **Machine Learning:** scikit-learn (`TransformedTargetRegressor`, `Pipeline`, `ColumnTransformer`, `RandomForestRegressor`, `RidgeCV`, `ElasticNetCV`)
- **Web Application:** Streamlit
- **Data Engineering:** pandas, NumPy, DVC
- **Data Visualization:** Matplotlib, Seaborn
- **Code Quality:** Ruff linter & formatter

---

## 👤 Author
- **Author:** Haadi Jafari
- **GitHub:** [@haadijafari](https://github.com/haadijafari)
- **Email:** [haadijafari2003@gmail.com](mailto:haadijafari2003@gmail.com)
