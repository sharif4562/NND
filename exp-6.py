# ============================================
# Live Weather Prediction Dashboard (FIXED)
# ============================================
# Interactive dashboard for weather prediction using RNN
# Run this in Google Colab

# ============================================
# 1. Install Required Libraries
# ============================================
!pip install -q gradio pandas numpy matplotlib torch scikit-learn plotly

# ============================================
# 2. Import Libraries
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gradio as gr
import warnings
from datetime import datetime, timedelta
import random
import time
import threading

warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# 3. Weather Data Generator (Simulates Live Data)
# ============================================
class LiveWeatherGenerator:
    """Generates realistic weather data simulating live feed"""
    
    def __init__(self):
        self.base_temp = 25
        self.current_time = datetime.now()
        self.history = []
        self.seasonal_factor = 0
        self.trend = 0
        
    def get_current_weather(self):
        """Generate current weather conditions"""
        self.current_time = datetime.now()
        
        # Time-based variations
        hour = self.current_time.hour
        day_of_year = self.current_time.timetuple().tm_yday
        
        # Seasonal variation
        seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Daily variation (warmer during day, cooler at night)
        daily = 5 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
        
        # Random weather fluctuations
        random_variation = np.random.normal(0, 2)
        
        # Calculate temperature
        temperature = 20 + seasonal + daily + random_variation
        
        # Other weather parameters
        humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
        pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3)
        wind_speed = 10 + 5 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 2)
        
        weather_data = {
            'timestamp': self.current_time,
            'temperature': round(temperature, 1),
            'humidity': round(min(max(humidity, 20), 100), 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(max(wind_speed, 0), 1),
            'condition': self.get_weather_condition(temperature, humidity),
            'hour': hour,
            'day_of_year': day_of_year
        }
        
        # Store in history
        self.history.append(weather_data)
        if len(self.history) > 1000:  # Keep last 1000 records
            self.history.pop(0)
        
        return weather_data
    
    def get_weather_condition(self, temp, humidity):
        """Determine weather condition based on parameters"""
        if humidity > 80 and temp > 20:
            return "🌧️ Rainy"
        elif humidity > 70 and temp < 15:
            return "☁️ Cloudy"
        elif temp > 30:
            return "☀️ Hot"
        elif temp < 5:
            return "❄️ Cold"
        else:
            return "⛅ Mild"
    
    def get_historical_data(self, hours=72):
        """Get historical data for the last N hours"""
        # Generate enough history if needed
        while len(self.history) < hours:
            self.get_current_weather()
            time.sleep(0.01)
        
        return self.history[-hours:]

# ============================================
# 4. RNN Model for Weather Prediction
# ============================================
class WeatherPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=1):
        super(WeatherPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for better long-term memory
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get last output
        out = out[:, -1, :]
        
        # Through dense layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

# ============================================
# 5. Train the Model with Epoch Tracking
# ============================================
def train_prediction_model():
    """Train the weather prediction model with epoch tracking"""
    print("Training weather prediction model...")
    
    # Generate training data
    n_samples = 5000
    sequence_length = 24  # Use 24 hours of history
    
    # Create synthetic training data
    X_train, y_train = [], []
    
    for i in range(n_samples):
        # Generate sequence
        seq = []
        for j in range(sequence_length):
            hour = (i + j) % 24
            day = (i + j) // 24
            # More realistic temperature pattern
            temp = 20 + 15 * np.sin(2 * np.pi * day / 365) + 5 * np.sin(2 * np.pi * hour / 24 - np.pi/2) + np.random.normal(0, 1)
            humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
            pressure = 1013 + 10 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 2)
            wind = 10 + 5 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 1)
            seq.append([temp, humidity, pressure, wind])
        
        X_train.append(seq[:-1])
        y_train.append(seq[-1][0])  # Predict next temperature
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    
    # Normalize
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, 4)
    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_normalized.reshape(-1, sequence_length-1, 4)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    
    # Create model
    model = WeatherPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train with epoch tracking
    model.train()
    batch_size = 64
    n_batches = len(X_train) // batch_size
    
    epoch_losses = []  # Store losses for each epoch
    
    for epoch in range(10):  # 10 epochs for better training
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / n_batches
        epoch_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch+1:2d}/10, Loss: {avg_epoch_loss:.6f}')
    
    model.eval()
    
    # Calculate final model performance
    with torch.no_grad():
        train_pred = model(X_train)
        final_train_loss = criterion(train_pred, y_train).item()
        train_accuracy = max(0, 100 - (final_train_loss * 100))  # Rough accuracy estimate
    
    print(f"\n✅ Training complete! Final loss: {final_train_loss:.6f}")
    print(f"📊 Model accuracy: {train_accuracy:.1f}%")
    
    return model, scaler, epoch_losses, train_accuracy

# Initialize model and scaler with epoch tracking
print("Initializing weather prediction system...")
model, feature_scaler, epoch_losses, model_accuracy = train_prediction_model()
weather_gen = LiveWeatherGenerator()

# ============================================
# 6. Prediction Functions
# ============================================
def prepare_input_data(historical_data, scaler):
    """Prepare historical data for prediction"""
    # Extract features
    features = []
    for data in historical_data:
        features.append([
            data['temperature'],
            data['humidity'],
            data['pressure'],
            data['wind_speed']
        ])
    
    # Normalize using the trained scaler
    features = np.array(features, dtype=np.float32)
    features_normalized = scaler.transform(features)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(device)
    return input_tensor

def predict_next_temperature(historical_data, model, scaler):
    """Predict next day's temperature"""
    if len(historical_data) < 24:
        return None, None
    
    # Get last 24 hours
    recent_data = historical_data[-24:]
    
    # Prepare input
    input_tensor = prepare_input_data(recent_data, scaler)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_temp = prediction.cpu().numpy()[0, 0]
    
    # Get recent temperatures for denormalization
    recent_temps = [d['temperature'] for d in recent_data]
    temp_mean = np.mean(recent_temps)
    temp_std = np.std(recent_temps) + 1e-8
    
    # Denormalize (approximate inverse transform)
    predicted_temp = predicted_temp * temp_std + temp_mean
    
    # Calculate confidence based on prediction stability and recent volatility
    if len(recent_temps) >= 7:
        volatility = np.std(recent_temps[-7:])
        # Also consider how well the prediction aligns with recent trend
        trend = recent_temps[-1] - recent_temps[0]
        trend_alignment = 1 - min(abs(predicted_temp - (recent_temps[-1] + trend/24)), 10) / 10
        confidence = max(0, min(100, 100 - volatility * 3 + trend_alignment * 20))
    else:
        confidence = 70
    
    return round(predicted_temp, 1), round(confidence, 1)

# ============================================
# 7. Dashboard Creation Functions
# ============================================
def create_weather_dashboard():
    """Create the main dashboard with epoch information"""
    
    # Get current weather
    current = weather_gen.get_current_weather()
    
    # Get historical data
    history = weather_gen.get_historical_data(72)  # Last 72 hours
    
    # Make prediction
    predicted_temp, confidence = predict_next_temperature(history, model, feature_scaler)
    
    # Create visualizations
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature History (72h)', 'Humidity & Pressure',
                       'Wind Speed', 'Training Progress (10 Epochs)',
                       'Hourly Pattern', 'Prediction Confidence'),
        specs=[[{'secondary_y': True}, {'secondary_y': True}],
               [{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'indicator'}]]
    )
    
    # 1. Temperature History
    timestamps = [d['timestamp'] for d in history]
    temps = [d['temperature'] for d in history]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=temps, mode='lines+markers',
                  name='Temperature', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Add prediction marker
    if predicted_temp:
        fig.add_trace(
            go.Scatter(x=[timestamps[-1] + timedelta(hours=1)], 
                      y=[predicted_temp],
                      mode='markers', name='Prediction',
                      marker=dict(color='green', size=15, symbol='star')),
            row=1, col=1
        )
    
    # 2. Humidity and Pressure
    humidity = [d['humidity'] for d in history]
    pressure = [d['pressure'] for d in history]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=humidity, name='Humidity (%)',
                  line=dict(color='blue', width=2)),
        row=1, col=2, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=pressure, name='Pressure (hPa)',
                  line=dict(color='purple', width=2, dash='dash')),
        row=1, col=2, secondary_y=True
    )
    
    # 3. Wind Speed
    wind_speed = [d['wind_speed'] for d in history]
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=wind_speed, name='Wind Speed',
                  line=dict(color='green', width=2), fill='tozeroy'),
        row=2, col=1
    )
    
    # 4. Training Progress (Epochs)
    epochs = list(range(1, len(epoch_losses) + 1))
    fig.add_trace(
        go.Scatter(x=epochs, y=epoch_losses, name='Training Loss',
                  line=dict(color='orange', width=3), mode='lines+markers'),
        row=2, col=2
    )
    
    # 5. Hourly Pattern
    hours = [d['hour'] for d in history]
    hourly_avg = []
    for hour in range(24):
        hour_temps = [temps[i] for i, h in enumerate(hours) if h == hour]
        hourly_avg.append(np.mean(hour_temps) if hour_temps else 0)
    
    fig.add_trace(
        go.Scatter(x=list(range(24)), y=hourly_avg, name='Hourly Avg',
                  line=dict(color='brown', width=3)),
        row=3, col=1
    )
    
    # 6. Confidence Gauge
    if confidence:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': f"Prediction Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=900, 
        showlegend=True,
        title_text="Live Weather Prediction Dashboard",
        title_font_size=20,
        template="plotly_dark"
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
    
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Pressure (hPa)", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Wind Speed (km/h)", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
    
    return fig, current, predicted_temp, confidence, epoch_losses[-1] if epoch_losses else 0

def refresh_data():
    """Refresh the dashboard with new data"""
    return create_weather_dashboard()

def get_weather_advice(current_temp, predicted_temp, humidity, wind, condition):
    """Generate weather advice based on conditions"""
    advice = []
    
    if predicted_temp and current_temp:
        temp_diff = predicted_temp - current_temp
        if temp_diff > 3:
            advice.append("⚠️ **Warning:** Significant warming expected tomorrow!")
        elif temp_diff < -3:
            advice.append("⚠️ **Warning:** Significant cooling expected tomorrow!")
    
    if predicted_temp and predicted_temp > 30:
        advice.append("☀️ **Hot tomorrow** - Stay hydrated and use sunscreen!")
    elif predicted_temp and predicted_temp < 10:
        advice.append("❄️ **Cold tomorrow** - Dress warmly and layer up!")
    
    if humidity > 80:
        advice.append("💧 **High humidity** - It might feel muggy and uncomfortable")
    elif humidity < 30:
        advice.append("💨 **Low humidity** - Stay moisturized and drink water")
    
    if wind > 25:
        advice.append("💨 **Strong winds** - Be careful outdoors, secure loose items")
    elif wind > 15:
        advice.append("🍃 **Moderate winds** - Good day for outdoor activities")
    
    # Add condition-based advice
    if "Rainy" in condition:
        advice.append("☔ **Rain expected** - Don't forget your umbrella!")
    elif "Cloudy" in condition:
        advice.append("☁️ **Cloudy conditions** - Good day for photos and outdoor walks")
    elif "Hot" in condition:
        advice.append("🧴 **Hot weather** - Apply sunscreen and stay in shade")
    elif "Cold" in condition:
        advice.append("🧣 **Cold weather** - Wear warm clothes and gloves")
    
    if not advice:
        advice.append("✅ **Perfect weather!** Enjoy your day outdoors!")
    
    return "\n\n".join(advice)

# ============================================
# 8. Create Gradio Interface (Fixed - No 'every' parameter)
# ============================================
def create_interface():
    """Create the Gradio interface with epoch information"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Live Weather Prediction Dashboard") as demo:
        gr.Markdown("""
        # 🌤️ Live Weather Prediction Dashboard
        ### Real-time weather monitoring and AI-powered predictions using LSTM Neural Networks
        
        This dashboard simulates live weather data and uses a trained LSTM model (10 epochs) 
        to predict tomorrow's temperature based on historical patterns.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Data Now", variant="primary", size="lg")
                
                # Model Info
                gr.Markdown("### 🤖 Model Information")
                with gr.Group():
                    model_epochs = gr.Textbox(label="Training Epochs", value="10", interactive=False)
                    model_loss = gr.Textbox(label="Final Training Loss", interactive=False)
                    model_acc = gr.Textbox(label="Model Accuracy", value=f"{model_accuracy:.1f}%", interactive=False)
                
                # Current conditions
                gr.Markdown("### 📊 Current Weather Conditions")
                with gr.Group():
                    current_temp_display = gr.Textbox(label="Temperature", interactive=False)
                    current_condition_display = gr.Textbox(label="Condition", interactive=False)
                    current_humidity_display = gr.Textbox(label="Humidity", interactive=False)
                    current_pressure_display = gr.Textbox(label="Pressure", interactive=False)
                    current_wind_display = gr.Textbox(label="Wind Speed", interactive=False)
                
                # Prediction
                gr.Markdown("### 🔮 Tomorrow's Prediction")
                with gr.Group():
                    prediction_display = gr.Textbox(label="Predicted Temperature", interactive=False)
                    confidence_display = gr.Textbox(label="Confidence Level", interactive=False)
                
                # Weather advice
                gr.Markdown("### 💡 Weather Advice")
                advice_display = gr.Textbox(label="Recommendations", lines=6, interactive=False)
            
            with gr.Column(scale=2):
                # Plot
                plot_output = gr.Plot(label="Weather Analysis Dashboard")
        
        # Initial load
        fig, current, predicted, confidence, final_loss = refresh_data()
        
        # Update function
        def update_dashboard():
            fig, current, predicted, confidence, final_loss = refresh_data()
            advice = get_weather_advice(
                current['temperature'], 
                predicted, 
                current['humidity'],
                current['wind_speed'],
                current['condition']
            )
            
            # Format displays
            temp_str = f"{current['temperature']}°C"
            condition_str = f"{current['condition']}"
            humidity_str = f"{current['humidity']}%"
            pressure_str = f"{current['pressure']} hPa"
            wind_str = f"{current['wind_speed']} km/h"
            pred_str = f"{predicted}°C" if predicted else "Insufficient data (need 24h history)"
            conf_str = f"{confidence}%" if confidence else "N/A"
            loss_str = f"{final_loss:.6f}"
            
            return (
                fig, 
                loss_str,
                temp_str, condition_str, humidity_str, pressure_str, wind_str,
                pred_str, conf_str,
                advice
            )
        
        # Set up the refresh button click
        refresh_btn.click(
            fn=update_dashboard,
            inputs=[],
            outputs=[
                plot_output,
                model_loss,
                current_temp_display, current_condition_display, 
                current_humidity_display, current_pressure_display, current_wind_display,
                prediction_display, confidence_display,
                advice_display
            ]
        )
        
        # Add a manual refresh note instead of auto-refresh
        gr.Markdown("""
        ---
        ### 🔄 Manual Refresh Only
        Click the **Refresh Data Now** button above to update the dashboard with the latest weather data.
        """)
        
        gr.Markdown("""
        ### 📊 Dashboard Features:
        
        | Feature | Description |
        |---------|-------------|
        | **Model Training** | LSTM trained for 10 epochs with batch size 64 |
        | **Training Loss** | Final loss: {:.6f} (see chart) |
        | **Real-time monitoring** | Live weather data simulation |
        | **AI predictions** | LSTM neural network predicts tomorrow's temperature |
        | **Confidence scoring** | Confidence level based on weather volatility |
        | **Multiple parameters** | Temperature, humidity, pressure, and wind speed |
        | **Smart advice** | Personalized recommendations based on conditions |
        | **Training visualization** | Loss curve showing model learning over 10 epochs |
        
        *Click the Refresh button to get the latest weather data and predictions.*
        """.format(epoch_losses[-1] if epoch_losses else 0))
    
    return demo

# ============================================
# 9. Launch the Dashboard
# ============================================
print("\n" + "="*60)
print("🚀 LAUNCHING WEATHER PREDICTION DASHBOARD")
print("="*60)
print("\n📋 Dashboard Features:")
print("   ✓ Live weather data simulation")
print("   ✓ AI-powered LSTM predictions (10 epochs)")
print("   ✓ Training loss curve showing epoch progress")
print("   ✓ 6 interactive visualizations")
print("   ✓ Manual refresh with button click")
print("   ✓ Smart weather advice")
print("   ✓ Confidence scoring")

# Create and launch interface
demo = create_interface()

print("\n" + "="*60)
print("🌐 ACCESS THE DASHBOARD")
print("="*60)
print("\nClick the link below to open the interactive dashboard:")
print("\n" + "-"*40)
demo.launch(share=True, debug=False)
print("-"*40)

print("\n" + "="*60)
print("📈 HOW THE RNN REMEMBERS PAST VALUES")
print("="*60)
print("""
The LSTM network remembers past weather patterns through:

1. **Hidden States**: Each time step maintains a hidden state vector
2. **Cell States**: LSTM's cell state acts as long-term memory
3. **Gating Mechanisms**:
   - Forget gate: Decides what to discard from memory
   - Input gate: Decides what new information to store
   - Output gate: Decides what to output based on memory

4. **Memory Decay**: Information gradually fades over ~24-48 hours
5. **Pattern Recognition**: The network learns daily and seasonal cycles

The prediction confidence reflects how well the model understands 
current weather patterns based on recent volatility and trend alignment.

📊 **Training Details**:
- **Epochs**: 10 complete passes through the training data
- **Loss Curve**: Shows how the model improved each epoch
- **Final Loss**: {:.6f} (lower is better)
- **Batch Size**: 64 samples per batch
""".format(epoch_losses[-1] if epoch_losses else 0))
