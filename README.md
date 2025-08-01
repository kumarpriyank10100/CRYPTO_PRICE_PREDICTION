# 📈 Cryptocurrency Price Prediction using LSTM!

Predict future Bitcoin prices using historical data and Long Short-Term Memory (LSTM) neural networks. This project leverages deep learning to model time-series data from Yahoo Finance and visualize predictions versus actual prices.

---

## 🚀 Features

- Historical BTC-USD price data from Yahoo Finance
- Data preprocessing and normalization
- Sequence generation for LSTM input
- LSTM neural network for price prediction
- Model training and evaluation
- Visualization of actual vs predicted prices

---

## 🧠 Technologies Used

- Python 🐍
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Yahoo Finance API via `yfinance`

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto-price-prediction-lstm.git
cd crypto-price-prediction-lstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

### 3. Run the Notebook

Open `CRYPTO_PRICE_PREDICTION.ipynb` in Jupyter Notebook or VS Code.

---

## 📊 Output

- The model predicts Bitcoin's closing price using previous 60-day windows.
- Plots show how well the model captures trends in BTC-USD price movements.

![Sample Plot](https://user-images.githubusercontent.com/yourplaceholder/btc_prediction_example.png)

---

## 📝 Notes

- This model is trained on past data and is not intended for real-time trading advice.
- Performance can be improved with more data, additional features (e.g., trading volume, news sentiment), and hyperparameter tuning.

---

## 🔮 Future Work

- Add multi-feature input (Open, High, Low, Volume)
- Predict other coins like ETH, XRP, etc.
- Use GRU or attention mechanisms
- Deploy as a Streamlit dashboard or Flask API

---

## 🙌 Acknowledgements

- Yahoo Finance (`yfinance`)
- TensorFlow/Keras team


---


