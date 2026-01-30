# ⚡ Electricity Demand Forecasting System

🚀 **Live App:** https://YOUR-STREAMLIT-LINK.streamlit.app  

---

## 📌 Overview

This project delivers a **production-style machine learning system** for short-term electricity demand forecasting using probabilistic modeling and real-time inference.

The application predicts future electricity consumption while quantifying uncertainty and detecting anomalous demand patterns — enabling data-driven operational planning similar to modern power utilities.

Unlike typical forecasting projects, this system integrates:

- Feature engineering with temporal lag structure  
- Probabilistic prediction intervals  
- Model calibration  
- Automated anomaly detection  
- Feature store architecture  
- Live inference pipeline  
- Interactive deployment  

👉 The result is a complete end-to-end ML system rather than a standalone model.

---

## 🎯 Objectives

- Forecast short-term electricity demand accurately  
- Provide uncertainty estimates for operational risk management  
- Detect abnormal consumption patterns  
- Enable interactive real-time predictions  
- Demonstrate production-grade ML architecture  

---

## 🧠 System Architecture
Feature Store → Model → Inference Pipeline → Interactive Dashboard


### Components:

**Feature Store**
- Engineered temporal features (lags, rolling statistics)
- Weather variables
- Calendar signals

**Model Layer**
- LightGBM regression models  
- Quantile regression for prediction intervals  

**Inference Engine**
- Ensures training-serving feature consistency  
- Generates calibrated probabilistic forecasts  

**Dashboard**
- Visualizes historical performance  
- Displays anomalies  
- Enables live demand simulation  

---

## ⚙️ Tech Stack

- **Python**
- **LightGBM**
- **Scikit-learn**
- **Pandas / NumPy**
- **Plotly**
- **Streamlit**
- **Parquet (feature store)**

---

## 🔍 Feature Engineering

The model captures temporal dynamics through:

- Lag features (1 to 288 intervals)
- Rolling mean and volatility indicators
- Hour-of-day patterns
- Weekend effects
- Weather-driven demand signals
- Circular encoding for wind direction

This structure allows the model to learn both short-term momentum and seasonal consumption behavior.

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| MAE | ~122 kW |
| SMAPE | ~3.08% |
| 95% Interval Coverage | ~95% |

The calibrated prediction intervals provide reliable uncertainty estimates — critical for real-world energy planning.

---

## 🚨 Anomaly Detection

The system automatically flags demand values falling outside calibrated prediction bands.

This enables rapid identification of:

- Unexpected demand spikes  
- Infrastructure stress signals  
- Data irregularities  

---

## 🔮 Live Forecasting

The deployed application supports **interactive real-time predictions**.

Users can adjust:

- Temperature  
- Humidity  
- Wind speed  
- Atmospheric pressure  
- Current demand  

The system instantly generates a probabilistic demand forecast.

👉 This simulates operational forecasting environments used by grid operators.

---

## 🏗️ Project Structure

├── app/
│ └── dashboard.py
│
├── src/
│ ├── inference.py
│ └── init.py
│
├── artifacts/
│ ├── models
│ ├── calibration
│ └── feature_store.parquet
│
└── requirements.txt


---

## ▶️ Running Locally

```bash
git clone https://github.com/YOUR-REPO.git
cd electricity-demand-forecasting

pip install -r requirements.txt

streamlit run app/dashboard.py

---

## 🌐 Live Application

The project is fully deployed and accessible publicly:

👉 https://electricity-demand-forecasting-anomaly-monitoring-sms6yh6s2akc.streamlit.app/

Experience the system in action — explore historical forecasts, visualize anomalies, and generate real-time electricity demand predictions through the interactive dashboard.

---

## 👤 Author

**Amit Meena**  
B.Sc. Computer Science  

If you found this project interesting or would like to discuss machine learning, forecasting systems, or applied AI, feel free to connect!


## 🧠 System Architecture

