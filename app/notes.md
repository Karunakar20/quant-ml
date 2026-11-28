ctual data test completed using PostgreSQL:

Symbol: RELIANCE.NS
Daily candles: 1,455
Date range: 2020-12-31 to 2026-11-19
Valid 1-day forward returns: 1,454
Classification results
Feature set	Model	Accuracy	Balanced accuracy	Macro F1
All	Logistic Regression	37.28%	36.74%	36.38%
All	Random Forest	35.84%	36.09%	35.86%
All	XGBoost	37.63%	37.82%	37.26%
Technical	Logistic Regression	40.50%	39.91%	39.70%
Technical	Random Forest	36.56%	36.87%	36.51%
Technical	XGBoost	38.35%	38.59%	38.25%
XGBoost regression results
Feature set	MAE	RMSE	R²	Direction accuracy
All	0.01125	0.01549	-0.1018	48.39%
Technical	0.01092	0.01547	-0.0987	50.90%
