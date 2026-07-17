# MSFT Stock Price Prediction using PyTorch LSTM 📈🤖

An end-to-end Deep Learning project designed to predict Microsoft Corporation (MSFT) stock prices using Long Short-Term Memory (LSTM) neural networks. This project demonstrates the ability to handle sequential time-series data, build custom PyTorch architectures, and overcome real-world financial data challenges.

🔴 **LIVE API ENDPOINT:** https://msft-prediction-api.onrender.com/predict

## 🎥 Project Presentation


https://github.com/user-attachments/assets/721478e0-51f7-414e-a45e-3eb927557029


## 🧠 Project Overview
Predicting stock market trends is notoriously difficult due to the chaotic nature of financial data. Traditional models often fail to capture long-term dependencies. This project utilizes an **LSTM Network** built from scratch in PyTorch to analyze the last 20 days of MSFT stock history and predict the next day's closing price.

### Key Highlights:
* **Real-time Data Extraction:** Automated historical data fetching directly from Yahoo Finance via the `yfinance` API, with built-in cloud IP block bypass mechanisms.
* **Deep Learning Architecture:** Custom PyTorch LSTM designed and hyperparameter-tuned (64 hidden neurons, 250 epochs) to solve underfitting and effectively capture aggressive market rallies.
* **Recursive Future Forecasting:** Features an advanced 30-day recursive forecasting loop to simulate long-term market trends visually.
* **Cloud Deployment (CI/CD):** The trained model (`.pth`) and fitted scaler (`.pkl`) are packaged into a robust Flask REST API using a CPU-optimized PyTorch build, running live on a Gunicorn server via Render.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch (CPU-optimized for cloud)
* **API Development & Server:** Flask, Gunicorn
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning Tools:** Scikit-learn (MinMaxScaler)
* **Data Visualization:** Matplotlib
* **Financial Data API:** yfinance

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/batuefe-ozsac/Stock-Price-Prediction-PyTorch.git](https://github.com/batuefe-ozsac/Stock-Price-Prediction-PyTorch.git)
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API Server locally:**
   ```bash
   python app.py
   ```
   *(Access `http://127.0.0.1:5000/predict` in your browser to see the live JSON response.)*

4. **Explore the Data & Forecast:** Open `stock_prediction.ipynb` in VS Code or Jupyter Notebook to run the cells sequentially and visualize the 30-day future forecast plots.

## 📊 Results & Insights
The model successfully learned the underlying mathematical patterns of the MSFT stock without simply memorizing the training data. During the testing phase (on unseen data), the model demonstrated a strong capability to track market trends, effectively catching sudden drops and aggressive bullish rallies in the tech sector. 

*(Future implementation: Exploring Multivariate LSTMs by adding trade volume and sentiment analysis to further reduce the error margin in volatile periods.)*
