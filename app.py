from flask import Flask, jsonify
import torch
import torch.nn as nn
import pickle
import numpy as np
import yfinance as yf

# 1. Start the Web Server
app = Flask(__name__)

# 2. Define the Neural Network Architecture (Required to load the brain)
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# 3. Load the Saved Brain and Translator
model = LSTMModel()
model.load_state_dict(torch.load('msft_lstm_model.pth'))
model.eval()

with open('msft_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 4. Create the API Endpoint
@app.route('/predict', methods=['GET'])
def predict_tomorrow():
    try:
        # Fetch the real-time last 20 days of MSFT using the more robust Ticker module
        msft = yf.Ticker("MSFT")
        data = msft.history(period="1mo")
        
        # Güvenlik Kontrolü: Yahoo Finance veriyi engellerse sistemi çökertme, uyarı ver
        if data.empty:
            return jsonify({
                "status": "error", 
                "message": "Cloud IP blocked by Yahoo Finance or data is unavailable. Please try again later."
            })

        last_20_days = data['Close'].values[-20:].reshape(-1, 1)
        
        # Translate to [-1, 1] and convert to PyTorch Tensor
        scaled_data = scaler.transform(last_20_days)
        tensor_data = torch.FloatTensor(scaled_data).unsqueeze(0)
        
        # Make Prediction
        with torch.no_grad():
            prediction = model(tensor_data).numpy()
            
        # Translate back to USD
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        return jsonify({
            "status": "success",
            "stock": "MSFT",
            "predicted_price_usd": round(float(predicted_price), 2)
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 5. Run the Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)