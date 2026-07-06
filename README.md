# MSFT Stock Price Prediction using PyTorch LSTM 📈🤖

An end-to-end Deep Learning project designed to predict Microsoft Corporation (MSFT) stock prices using Long Short-Term Memory (LSTM) neural networks. This project demonstrates the ability to handle sequential time-series data, build custom PyTorch architectures, and overcome real-world financial data challenges.

## 🧠 Project Overview
Predicting stock market trends is notoriously difficult due to the chaotic nature of financial data. Traditional models often fail to capture long-term dependencies. This project utilizes an **LSTM Network** built from scratch in PyTorch to analyze the last 20 days of MSFT stock history and predict the next day's closing price.

### Key Highlights:
* **Real-time Data Extraction:** Automated historical data fetching directly from Yahoo Finance via the `yfinance` API.
* **Data Preprocessing:** Utilized `MinMaxScaler` for normalizing values between [-1, 1] to optimize the neural network's gradient descent, along with robust sliding-window sequence generation.
* **Hyperparameter Optimization:** Initially tested with a small capacity, the model was iteratively upgraded (to 64 hidden neurons and 250 epochs) to solve underfitting and effectively capture aggressive market rallies without overfitting (perfect fit).
* **Dynamic Checkpointing:** Implemented automated saving of trained PyTorch model weights (`.pth`) and the fitted scaler (`.pkl`) for future deployment and inference.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning Tools:** Scikit-learn (MinMaxScaler)
* **Data Visualization:** Matplotlib
* **Financial Data API:** yfinance

## 🚀 How to Run Locally

1. **Clone the repository:**
   `git clone https://github.com/batuefe-ozsac/Stock-Price-Prediction-PyTorch.git`

2. **Install the required dependencies:**
   `pip install torch numpy pandas matplotlib scikit-learn yfinance`

3. **Run the Notebook:** Open `stock_prediction.ipynb` in VS Code or Jupyter Notebook and run the cells sequentially.

## 📊 Results & Insights
The model successfully learned the underlying mathematical patterns of the MSFT stock without simply memorizing the training data. During the testing phase (on unseen data), the model demonstrated a strong capability to track market trends, effectively catching sudden drops and aggressive bullish rallies in the tech sector. 

*(Future implementation: Exploring Multivariate LSTMs by adding trade volume and sentiment analysis to further reduce the error margin in volatile periods.)*