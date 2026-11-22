## Project Folder Structure

Quant ML follows a **Modular Monolith Architecture** to maintain a clean separation of concerns, improve maintainability, and support future scalability.

Instead of introducing microservices at the initial stage, the application is organized into independent modules with clearly defined responsibilities. This approach keeps development and deployment simple while allowing individual components to evolve independently.

### Directory Structure

```text
quant-ml/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── stocks.py
│   │       ├── predictions.py
│   │       ├── backtests.py
│   │       └── models.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── data/
│   │   ├── providers/
│   │   │   ├── fyers.py
│   │   │   ├── nse.py
│   │   │   └── yahoo.py
│   │   ├── ingestion.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── technical.py
│   │   ├── market_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── ml/
│   │   ├── datasets.py
│   │   ├── targets.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── calibration.py
│   │   └── models/
│   │       ├── xgboost_model.py
│   │       ├── random_forest.py
│   │       └── lstm_model.py
│   │
│   ├── backtesting/
│   │   ├── engine.py
│   │   ├── execution.py
│   │   ├── risk.py
│   │   ├── strategies.py
│   │   └── metrics.py
│   │
│   ├── services/
│   │   ├── prediction_service.py
│   │   └── screening_service.py
│   │
│   ├── database/
│   │   ├── models.py
│   │   ├── session.py
│   │   └── repositories/
│   │
│   └── tasks/
│       ├── ingestion_tasks.py
│       └── training_tasks.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_xgboost.ipynb
│   └── 04_lstm.ipynb
│
├── tests/
│
├── scripts/
│
├── migrations/
│
├── artifacts/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Eight-Week Implementation Roadmap

The project will be developed over **eight weeks**, following an incremental implementation approach.

**Estimated daily commitment:** 2–3 hours

Each phase focuses on a specific engineering or machine learning component and produces a working deliverable before moving to the next phase.

### Week 1 – Data Engineering and Collection

**Objectives:**
- Set up the Python development environment.
- Configure PostgreSQL and database connectivity.
- Integrate the FYERS Historical Data API.
- Collect historical OHLCV data for NIFTY 50.
- Implement data ingestion and storage pipelines.
- Validate missing candles, duplicate records, and invalid values.

**Deliverable:** Automated historical market data collection and storage pipeline.

### Week 2 – Exploratory Data Analysis and Feature Engineering

**Objectives:**
- Perform exploratory data analysis (EDA) on historical market data.
- Analyze stock returns, volatility, volume, and price movements.
- Implement technical indicators:
  - RSI
  - EMA
  - MACD
  - ATR
  - Bollinger Bands
- Generate market-relative features using NIFTY 50 and relevant sector indices.
- Create forward-return targets for multiple prediction horizons.
- Prepare the initial ML-ready dataset.

**Deliverable:** Reusable feature engineering pipeline and ML-ready dataset.

### Week 3 – Machine Learning Model Development

**Objectives:**
- Establish a Logistic Regression baseline model.
- Implement Random Forest classification.
- Implement XGBoost classification.
- Implement XGBoost regression.
- Experiment with different feature combinations.
- Compare model performance using appropriate evaluation metrics.

**Deliverable:** Initial trained ML models for stock trend prediction and return estimation.

### Week 4 – Model Evaluation and Optimization

**Objectives:**
- Implement time-series-aware walk-forward validation.
- Perform hyperparameter tuning.
- Evaluate classification and regression performance.
- Implement probability calibration.
- Analyze feature importance.
- Use SHAP for model interpretability.
- Evaluate model performance across different market conditions.

**Deliverable:** Optimized and evaluated ML models with documented performance results.

### Week 5 – Backtesting and Risk Management

**Objectives:**
- Develop a historical backtesting engine.
- Implement trading strategies based on model predictions.
- Define entry and exit conditions.
- Implement stop-loss and trailing stop-loss mechanisms.
- Implement position sizing and capital allocation.
- Include brokerage, taxes, slippage, and other applicable transaction costs.
- Calculate risk-adjusted performance metrics.
- Compare strategy performance against relevant benchmarks.

**Deliverable:** Functional backtesting engine with historical performance reports.

### Week 6 – Deep Learning Implementation

**Objectives:**
- Understand neural network fundamentals.
- Implement an LSTM-based time-series prediction model.
- Prepare sequential datasets for deep learning.
- Experiment with different sequence lengths and network architectures.
- Compare LSTM performance against XGBoost.
- Evaluate models across different market regimes.

**Deliverable:** Trained LSTM model and comparative ML versus DL evaluation report.

### Week 7 – Backend Development and ML Integration

**Objectives:**
- Develop REST APIs using FastAPI.
- Integrate PostgreSQL with SQLAlchemy.
- Implement stock screening and prediction services.
- Integrate Redis for caching and background task coordination.
- Implement Celery for asynchronous data ingestion and model training.
- Integrate MLflow for experiment tracking and model management.
- Develop API endpoints for predictions, backtesting, and model information.

**Deliverable:** Functional FastAPI backend integrated with the machine learning pipeline.

### Week 8 – Production Readiness and Deployment

**Objectives:**
- Containerize the application using Docker.
- Configure multi-container deployment using Docker Compose.
- Implement automated testing using Pytest.
- Add application logging and error handling.
- Implement model performance and data quality monitoring.
- Develop a dashboard for stock screening, predictions, and backtesting results.
- Prepare API documentation.
- Publish project documentation and source code on GitHub.
- Deploy the application to a suitable hosting environment.

**Deliverable:** End-to-end BharatQuant ML application with backend APIs, ML models, backtesting, dashboard, testing, and deployment documentation.

---

### Final Project Milestones

| Timeline | Major Milestone |
|---|---|
| Week 1 | Historical market data pipeline |
| Week 2 | Feature engineering and dataset preparation |
| Week 3 | Initial machine learning models |
| Week 4 | Model evaluation and optimization |
| Week 5 | Backtesting and risk management |
| Week 6 | Deep learning implementation |
| Week 7 | FastAPI and ML integration |
| Week 8 | Production readiness and deployment |

**Important:** All machine learning predictions and backtesting results are experimental. Historical performance does not guarantee future returns. The system is intended for research, analysis, and educational purposes rather than guaranteed trading recommendations.