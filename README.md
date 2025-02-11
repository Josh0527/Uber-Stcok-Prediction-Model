# Uber-Stock-Prediction-Model
# Uber Stock Price Prediction using LSTM

This project demonstrates the use of Long Short-Term Memory (LSTM) networks to predict the stock price of Uber. The model is trained on historical stock data and evaluated on a test set to assess its predictive performance.

# Project Overview
The project involves the following steps:

1. **Data Acquisition**: The dataset used in this project contains historical stock prices for Uber, including the date, closing price, and other relevant information.
2. **Data Preprocessing**: The data is cleaned and preprocessed to handle missing values, convert dates to the appropriate format, and scale the data using MinMaxScaler.
3. **Model Building**: An LSTM model is constructed with two LSTM layers and two Dense layers. The model is compiled with the Adam optimizer and mean squared error loss function.
4. **Model Training**: The model is trained on a portion of the data (training set) for a specified number of epochs and batch size.
5. **Model Evaluation**: The trained model is evaluated on a separate portion of the data (test set) to assess its performance using metrics like mean squared error.
6. **Prediction Visualization**: The predicted and actual stock prices are plotted to visualize the model's predictions.

# Dependencies
- Python 3.
- pandas
- seaborn
- matplotlib
- numpy
- kagglehub
- pytz
- statsmodels
- scikit-learn
- TensorFlow/Keras

# Usage
1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the dataset and place it in the project directory.
4. Run the Jupyter notebook `Uber_Stock_Prediction.ipynb` to execute the code.

# Results

The model's performance is evaluated using the test loss and visualized through a plot comparing the actual and predicted stock prices. You can find the results in the Jupyter notebook.

# Future Work

- Experiment with different model architectures and hyperparameters to improve performance.
- Incorporate additional features, such as news sentiment or economic indicators, to enhance the model's predictive power.
- Deploy the model for real-time stock price prediction.


# Disclaimer

This project is for educational purposes only and should not be considered financial advice. The stock market is inherently unpredictable, and past performance is not indicative of future results.
