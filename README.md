🪙 **Bitcoin Price Prediction Using SVM**

This project is focused on predicting Bitcoin prices using a machine learning technique called Support Vector Regression (SVR). It uses historical Bitcoin price data, processes it, and trains a model to predict future prices.


📌 **Objective**

The main goal is to build a machine learning model that can:

1. Understand patterns in Bitcoin price trends

2. Predict future prices based on past values

3. Provide visual insights through graphs and model metrics


📂 **Project Organization**

The project is organized into two main folders:

1. Notebooks/ – Contains the Jupyter notebooks used for data processing and modeling

2. data/ – Contains the raw historical Bitcoin price datasets


📊 **Data Used**

The project uses two CSV files:

1. bitcoin_price.csv

2. bitcoin_dataset.csv

Each file contains historical price data, including:

1. Date and Time

2. Opening and Closing prices

3. Highest and Lowest prices
   
4. Volume of Bitcoin traded


🔍 **Methodology**

The project follows these main steps:

1. Data Preprocessing – Cleaning, formatting, and preparing the dataset

2. Feature Selection – Choosing which price features to use

3. Modeling – Using the SVR (Support Vector Regression) algorithm

4. Evaluation – Using metrics like R² Score and Mean Squared Error

5. Visualization – Graphs to compare actual vs predicted prices

🧠 **Machine Learning Model**

The model used in this project is:

1. Support Vector Regression (SVR)

2. Kernel: Radial Basis Function (RBF)

This model excels at handling non-linear patterns in time-series data, such as cryptocurrency prices.

📈 **Results**

1. The SVR model was able to learn the patterns in Bitcoin price movement

2. Graphs in the notebook show how close the predictions are to the actual values

3. Performance was evaluated using the R² score and MSE

✅ **Tools and Technologies**

1. Python

2. Jupyter Notebook

3. Scikit-learn (for ML modeling)

4. Pandas and NumPy (for data handling)

5. Matplotlib (for plotting graphs)

📘 **Credits**

This project was created as a mini machine learning case study. It is useful for students or developers interested in financial data prediction using traditional ML techniques.

