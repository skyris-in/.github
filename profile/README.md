# Skyris - IoT + ML Cloudburst Prediction System

**Real-time cloudburst prediction powered by IoT sensor networks and ensemble machine learning**

---

# Overview

**Skyris** is an end-to-end IoT + Machine Learning framework for **real-time cloudburst prediction and early warning**. The system combines a network of environmental sensors deployed on an ESP8266 microcontroller with an ensemble of tabular and vision-based ML models served via a FastAPI backend, all visualized through a modern Next.js web dashboard.

Cloudbursts — sudden, intense rainfall events — are responsible for devastating flash floods, landslides, and loss of life, particularly in mountainous regions. Skyris addresses this by:

1. **Collecting real-time atmospheric data** from IoT sensors (temperature, humidity, pressure, rainfall intensity, light/optical thickness, distance/cloud base height).
2. **Feeding sensor data into an ensemble of ML models** (XGBoost, Random Forest, SVM) for tabular prediction, alongside CNN and DenseNet vision models for cloud imagery analysis.
3. **Generating a consensus prediction** using majority voting across all models with rolling-window aggregation.
4. **Displaying live predictions, sensor readings, and weather data** on an interactive dashboard with real-time charts, maps, and alert systems.

---

# ML Backend Architecture

![ML Backend Architecture](https://github.com/PrakharSinghOnGit/skyris-graphethon/blob/main/asset/MLBackend.png)

---

# System Architecture (To be Changed)

```


┌──────────────────────────────────────────────────────────────────────┐
│                        SKYRIS ARCHITECTURE                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     Serial/USB      ┌──────────────────────┐       │
│  │  ESP8266     │ ──────────────────► │  Python Serial       │       │
│  │  + Sensors   │    CSV Data         │  Bridge (api.py)     │       │
│  │  (IoT Node)  │                     └──────────┬───────────┘       │
│  └──────────────┘                                │                   │
│                                                  │ HTTP POST         │
│  ┌─────────────┐     HTTP POST                  ▼                    │
│  │  Camera     │ ─────────────►  ┌──────────────────────────┐        │
│  │  Module     │   /image/       │   FastAPI ML Server      │        │
│  └─────────────┘                 │   (server.py)            │        │
│                                  │                          │        │
│                                  │  ┌────────────────────┐  │        │
│                                  │  │ Tabular Models     │  │        │
│                                  │  │  • XGBoost         │  │        │
│                                  │  │  • Random Forest   │  │        │
│                                  │  │  • SVM (SVC)       │  │        │
│                                  │  └────────────────────┘  │        │
│                                  │  ┌────────────────────┐  │        │
│                                  │  │ Vision Models      │  │        │
│                                  │  │  • CNN             │  │        │
│                                  │  │  • DenseNet        │  │        │
│                                  │  └────────────────────┘  │        │
│                                  │                          │        │
│                                  │  Rolling Window (5)      │        │
│                                  │  Majority Voting         │        │
│                                  └────────────┬─────────────┘        │
│                                               │                      │
│                                               │ GET /latest-result/  │
│                                               ▼                      │
│                                  ┌──────────────────────────┐        │
│                                  │   Next.js 15 Frontend    │        │
│                                  │   (React 19 + TypeScript)│        │
│                                  │                          │        │
│                                  │  • Real-time Dashboard   │        │
│                                  │  • Sensor Charts         │        │
│                                  │  • Weather Integration   │        │
│                                  │  • Interactive Map       │        │
│                                  │  • Alert System          │        │
│                                  └──────────────────────────┘        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## IoT Hardware & Sensors

### Microcontroller

| Component         | Specification                     |
| ----------------- | --------------------------------- |
| **Board**         | ESP8266 (NodeMCU / Wemos D1 Mini) |
| **Clock**         | 80 MHz / 160 MHz                  |
| **Flash**         | 4 MB                              |
| **Communication** | USB Serial @ 115200 baud          |
| **I2C Pins**      | D2 (SDA), D1 (SCL)                |

### Sensor Array

| Sensor                     | Model         | Measures                                | Interface              | Pin(s)               | Output Range                   |
| -------------------------- | ------------- | --------------------------------------- | ---------------------- | -------------------- | ------------------------------ |
| **Temperature & Humidity** | DHT11         | Temperature (°C), Relative Humidity (%) | Digital (One-Wire)     | D4 (GPIO2)           | Temp: 0–50°C, Humidity: 20–90% |
| **Barometric Pressure**    | BMP280        | Atmospheric Pressure (hPa)              | I2C (0x76)             | D2/D1                | 300–1100 hPa                   |
| **Light Intensity**        | BH1750        | Illuminance / Optical Thickness (lux)   | I2C (0x23 or 0x5C)     | D2/D1                | 1–65535 lux                    |
| **Rain Sensor**            | YL-83 / FC-37 | Rainfall Intensity (analog)             | Analog                 | A0                   | 0–1024 (inverted)              |
| **Ultrasonic Distance**    | HC-SR04       | Cloud Base Height approximation (cm)    | Digital (Trigger/Echo) | D6 (Trig), D7 (Echo) | 2–400 cm                       |

---

## Data Output Format (CSV over Serial)

The Arduino firmware transmits sensor readings every **5 seconds** as CSV:

```
Cloud_Top_Height, Cloud_Base_Height, Optical_Thickness, Rainfall, Humidity, Temperature, Pressure
```

- **Cloud_Top_Height**: Encoded binary flag (0 or 1) based on rainfall intensity threshold (>100 → 1).
- **Cloud_Base_Height**: Ultrasonic distance measurement (cm).
- **Optical_Thickness**: Light level from BH1750 (lux, used as optical thickness approximation).
- **Rainfall**: Inverted analog rain sensor value (1024 - raw).
- **Humidity**: DHT11 humidity reading (%).
- **Temperature**: DHT11 temperature reading (°C).
- **Pressure**: BMP280 pressure reading (hPa).

---

## ML Models & Pipeline

### Tabular Models (Numerical Sensor Data)

Trained on a **10,000-record cloudburst dataset** with 7 features and binary classification.

| Model                   | File                            | Type              | Description                       |
| ----------------------- | ------------------------------- | ----------------- | --------------------------------- |
| **XGBoost**             | `xgboost_model.pkl`             | Gradient Boosting | High-accuracy ensemble tree model |
| **Random Forest**       | `random_forest_model.pkl`       | Bagging Ensemble  | Robust multi-tree classifier      |
| **SVM (SVC)**           | `svc_model.pkl`                 | Support Vector    | Kernel-based binary classifier    |
| **Logistic Regression** | `logistic_regression_model.pkl` | Linear            | Baseline probabilistic classifier |

### Vision Models (Cloud Imagery)

| Model        | File                                    | Architecture               | Input Size  | Description                                  |
| ------------ | --------------------------------------- | -------------------------- | ----------- | -------------------------------------------- |
| **CNN**      | `cnn_binary_classification_model.keras` | Custom CNN                 | 128×128 RGB | Binary cloud classification                  |
| **DenseNet** | `DenseNet_model.keras`                  | DenseNet Transfer Learning | 128×128 RGB | Dense connectivity pattern for feature reuse |

### Ensemble Prediction Pipeline

```
Sensor Data (every 5s)
        │
        ▼
┌──────────────────────┐
│ StandardScaler       │  (Fitted on training data)
│ Transform features   │
└──────────┬───────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  XGBoost  RF   SVC        ← Predict class + probability
     │     │     │
     └─────┼─────┘
           │
           ▼
  Rolling Window (size=5)   ← Aggregate last 5 predictions
           │
           ├──────────────── Vision Models (CNN + DenseNet)
           │                 (if image received in window)
           ▼
  Majority Voting            ← All model votes combined
           │
           ▼
  Final Prediction           ← Class (0/1) + Probability
```

- **Rolling Window**: Aggregates 5 consecutive predictions before computing final result.
- **Majority Voting**: Tabular model votes + vision model votes → majority decides final class.
- **Probability**: Average of per-model cloudburst probabilities (class 1) across the window.

### Training Notebooks

| Notebook                   | Location                                                       |
| -------------------------- | -------------------------------------------------------------- |
| XGBoost Classifier         | `ml_backend/num_model/notebooks/XGBoostClassifier.ipynb`       |
| Random Forest Classifier   | `ml_backend/num_model/notebooks/RandomForestClassifier.ipynb`  |
| SVM (SVC) Classifier       | `ml_backend/num_model/notebooks/SupportVectorClassifier.ipynb` |
| Logistic Regression        | `ml_backend/num_model/notebooks/Logistic_Regression.ipynb`     |
| CNN Binary Classification  | `ml_backend/vision_model/notebooks/CNN.ipynb`                  |
| DenseNet Transfer Learning | `ml_backend/vision_model/notebooks/DenseNet.ipynb`             |

---

## Tech Stack

| Technology             | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| **FastAPI**            | High-performance REST API framework                  |
| **Uvicorn**            | ASGI server                                          |
| **scikit-learn**       | StandardScaler, SVC, Random Forest, train/test split |
| **XGBoost**            | Gradient boosting classifier                         |
| **TensorFlow / Keras** | CNN and DenseNet vision models                       |
| **Joblib**             | Model serialization                                  |
| **Pandas / NumPy**     | Data manipulation                                    |
| **Pillow**             | Image preprocessing                                  |

### IoT / Embedded

| Technology            | Purpose                                        |
| --------------------- | ---------------------------------------------- |
| **Arduino IDE**       | Firmware development for ESP8266               |
| **ESP8266 (NodeMCU)** | Microcontroller with WiFi                      |
| **PySerial**          | Serial communication bridge (Arduino → Python) |

---

# Dataset

The system is trained on a **10,000-record cloudburst prediction dataset** with the following composition:

| Source Type        |    Records | Cloudburst Rate | Description                      |
| ------------------ | ---------: | --------------: | -------------------------------- |
| Actual-based       |        160 |           97.5% | Real historical event variations |
| Extreme Synthetic  |      1,500 |           99.1% | Severe cloudburst conditions     |
| Moderate Synthetic |      1,500 |           99.1% | Moderate cloudburst conditions   |
| Marginal Synthetic |      3,500 |           53.7% | Borderline conditions            |
| Normal Synthetic   |      3,340 |           10.4% | Typical weather scenarios        |
| **Total**          | **10,000** |       **53.5%** | **Balanced classification**      |

### Feature Ranges

| Feature           | Unit          | Min    | Max    | Mean   |
| ----------------- | ------------- | ------ | ------ | ------ |
| Cloud Top Height  | meters        | 10,559 | 19,490 | 15,130 |
| Cloud Base Height | meters        | 784    | 6,132  | 2,294  |
| Optical Thickness | dimensionless | 13.9   | 104.9  | 57.2   |
| Rainfall          | mm/hour       | 113.9  | 639.3  | 230.4  |
| Humidity          | %             | 63.0   | 102.5  | 82.8   |
| Temperature       | °C            | 2.5    | 43.8   | 20.6   |
| Pressure          | hPa           | 800.7  | 1026.7 | 931.4  |

### Historical Events in Dataset

The dataset incorporates real-world cloudburst events including:

- **Kedarnath Disaster** (2013) — 326 mm rainfall, Uttarakhand, India
- **Mumbai Floods** (2005) — 944 mm rainfall, Mumbai, India
- **Ahr Valley Floods** (2021) — 150 mm/18hr, Germany
- **HP Solan** (2023) — 273 mm rainfall, Himachal Pradesh, India
- And 4 more documented events

---

## How It Works

### 1. Data Collection (IoT Layer)

The ESP8266 microcontroller reads data from all 5 sensors every 5 seconds, formats it as CSV, and transmits it over USB serial.

### 2. Data Ingestion (Bridge Layer)

The Python serial bridge (`api.py`) reads CSV lines from the Arduino, parses and validates them, then sends the data as JSON to the FastAPI ML server via HTTP POST.

### 3. Prediction (ML Layer)

The FastAPI server:

- Scales incoming features using a pre-fitted `StandardScaler`.
- Runs data through **3 tabular models** (XGBoost, Random Forest, SVC) to get class predictions and probabilities.
- If a cloud image is received, runs it through **2 vision models** (CNN, DenseNet).
- Maintains a **rolling window of 5 predictions** per model.
- When the window is full, performs **majority voting** across all model votes to produce a final consensus prediction and an averaged probability.

### 4. Visualization (Frontend Layer)

The Next.js dashboard:

- **Polls the ML server every 5 seconds** for the latest prediction via `GET /latest-result/`.
- Displays real-time line charts for each sensor reading (Cloud Base Height, Optical Thickness, Rainfall, Humidity, Temperature, Pressure).
- Shows a **radial probability gauge** for cloudburst risk level.
- Provides a **cloudburst warning banner** (Low / Medium / High risk).
- Integrates **OpenWeatherMap** for current weather conditions at the user's geolocation.
- Renders an **interactive Leaflet map** showing the user's location and risk zones.
- Includes a **Precautions page** with emergency contacts, do's and don'ts, FAQs, and resources.

### 5. Authentication (Security Layer)

- Supabase Auth with email/password (PKCE flow).
- Protected routes enforced via Next.js middleware.
- User profiles stored in Supabase PostgreSQL with automatic creation on sign-up (database trigger).

---

## Dashboard Features

| Feature                      | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| **Cloudburst Warning Card**  | Risk-level display (Low/Medium/High) with contextual messaging                     |
| **Radial Probability Gauge** | Visual consensus probability across all models                                     |
| **Sensor Line Charts**       | Real-time charts for each of 6 sensor readings with history                        |
| **Model Probabilities**      | Per-model prediction breakdown (XGBoost, RF, SVM)                                  |
| **Interactive Map**          | Leaflet map with user location, city center, and risk circles                      |
| **Weather Dashboard**        | Full OpenWeatherMap integration (temp, wind, humidity, visibility, sunrise/sunset) |
| **Precautions Page**         | Before/During/After cloudburst guidelines, emergency contacts, FAQs                |
| **Dark/Light Mode**          | Theme switching with `next-themes`                                                 |
| **Responsive Design**        | Mobile-first with collapsible sidebar                                              |
| **Live Status Indicator**    | Green/red connection status dot with manual refresh                                |

---
