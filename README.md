# hnco
In this project I implemented a model described in the book "Machine Learning for Asset Managers" by Lopez de Prado (2019) called "Nested Clustered Optimisation".  Some alterations to the original model were made and several backtesting techniques were applied in order to test the goodness of HNCO portfolio performance.

Here are descriptions of all files attached to the project:
1. NCO_data.csv contains preprocessed data for optimal weights computation. This data is used in both Backtesting.ipynb and full_NCO_code.ipynb.
2. F-F_Research_Data_Factors_weekly.csv contains annualized risk-free rates from july 2, 1926, to may 31, 2023. This data is used for Sharpe ratio calculations.
3. full_NCO_code.ipynb describes the algorithm of HNCO weights computation in details, where the training set is the entire NCO_data.csv dataset.
4. Backtesting.ipynb describes Walk-forward and Cross-validation backtesting procedures aimed to compare equal-weighted, Markowitz minvariance and HNCO portfolios based on NCO_data.csv dataset.
5. DP_functions.py contains all necessary functions for aforementioned scripts.
6. HDBSCAN_NCO_Course_paper.pdf is the course paper written by me dedicated to Lopez de Prado's NCO algorithm. It contains theoretical explanation of all concepts used in weights computation and results of NCO implementation in python.
