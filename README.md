# Olist Late Delivery Prediction

**Predicting late e-commerce deliveries at carrier handoff using historical order data.**

This machine-learning project uses the **Olist Brazilian E-Commerce** dataset to find orders that may arrive later than promised. It compares a simple model using information known at purchase time with models using extra information known when the seller hands the package to the carrier. The full analysis, feature engineering, model comparison, and test results are in [`notebooks/delivery_prediction.ipynb`](notebooks/delivery_prediction.ipynb).

## Dataset

The project uses the **Olist Brazilian E-Commerce Public Dataset** from Brazil. The notebook reads prepared order, item, product, payment, seller, customer, and location data from a **PostgreSQL database** after an upstream data pipeline / dbt transformation.

| Data source | Purpose |
| --- | --- |
| `analytics_core` tables | Order, item, product, seller, payment, and location information |
| `analytics_staging` tables | Original order timestamps and customer information |
| `late_delivery` target | `1` if an order arrived after its promised delivery date; otherwise `0` |

**Original dataset:** [Olist Brazilian E-Commerce Public Dataset (Kaggle)

## Methodology

1. **Load and check the data.** Query PostgreSQL and create one row per order. Remove impossible delivery dates and check the late-delivery labels.
2. **Explore delivery patterns.** Review the late-delivery rate, missing values, and changes over time.
3. **Create seller history without future information.** Use `merge_asof` to count deliveries and late orders completed before each carrier handoff. Use the training late rate when a seller has no history.
4. **Split by time, not randomly.** Use older orders for training, later orders for validation, and the newest orders for testing. Training and validation labels must be known before the next period starts.
5. **Build two feature sets.** Compare **16 purchase-time features** (such as price, freight, and product category) with **32 handoff-time features** (adding preparation time, remaining delivery time, approximate distance, payment details, and past seller performance).
6. **Compare models on validation data.** Test Logistic Regression, Random Forest, and CatBoost (depth 4 and 6). Choose the handoff model with the highest **Recall@15%**, using **Average Precision (PR-AUC)** to break ties.
7. **Evaluate on the untouched test set.** Score the purchase-time baseline and selected handoff model on exactly the same test orders.

**Why 15%?** The model scores *all* orders but selects only the 15% with the highest predicted risk for possible review. This is an **assumed review-capacity limit**, not a claim that 15% of orders are late.


### Validation — model selection

The **Handoff CatBoost (depth 6)** model was selected by its validation **Recall@15% of 61.7%**. The purchase-time Logistic Regression baseline had **42.2%** on the same validation set.

### Later test period — 14,610 orders

The test period begins on **June 22, 2018**. Only **4.28%** of these orders arrived late.

| Model | Recall@15% ↑ | Precision@15% ↑ | Lift@15% ↑ | Average Precision (PR-AUC) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Purchase-time Logistic Regression | 25.1% | 7.2% | 1.67× | 0.082 |
| **Handoff CatBoost (depth 6)** | **71.5%** | **20.4%** | **4.77×** | **0.515** |

Both models flagged **2,192 orders** (top 15%). The handoff model found **447 of 625** actual late deliveries, compared with **157** for the purchase-time baseline. That is a **46.4 percentage-point improvement** in Recall@15%.

The selected handoff queue also contained **1,745 orders that arrived on time**, and **178 late orders were missed**. The results are useful for prioritizing reviews, but they do **not** mean every flagged order will be late.

## Run locally

**Requirements:** Python **3.11** recommended, a running PostgreSQL database with the required dbt tables, and database credentials.

The raw Olist files and data warehouse setup are not included in this repository. The data preparation and transformation process can be found in the [Olist E-Commerce Analytics Pipeline](https://github.com/oraya-chai/Olist-E-Commerce-Analytics-Pipeline).


```bash
# Optional: create a virtual environment
python -m venv .venv

# Activate it: macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# Install Python packages
python -m pip install -r requirements.txt

# Open JupyterLab
python -m jupyter lab
```

Create a local **`.env`** file in the project folder with your own PostgreSQL connection details:

```dotenv
ML_DB_HOST=localhost
ML_DB_PORT=5432
ML_DB_USER=your_database_user
ML_DB_PASSWORD=your_database_password
ML_DB_NAME=your_database_name
```

## Repository contents

```text
.
├── README.md                        
├── requirements.txt               
├── .gitignore                 
└── notebooks/
    └── delivery_prediction.ipynb 
