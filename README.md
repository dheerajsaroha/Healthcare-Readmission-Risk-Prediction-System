# 🏥 Healthcare Readmission Risk Prediction System

End-to-end Machine Learning system to predict patient readmission risk using structured healthcare data.

This project demonstrates the complete ML lifecycle — data preprocessing, feature engineering, model training, evaluation, REST API deployment, browser-based UI, and Dockerized inference.

---

## 🚀 Features

- RandomForest-based classification model  
- Feature engineering on clinical attributes  
- ROC-AUC evaluation and class imbalance handling  
- Flask REST API for real-time predictions  
- Browser-based UI for interactive risk scoring  
- Dockerized full-stack deployment (API + UI)  
- Input validation and basic prediction monitoring  

---

## 🧠 Tech Stack

- Python 3.10  
- Scikit-learn  
- Pandas / NumPy  
- Flask  
- HTML / CSS (UI)  
- Docker  

---

## 📊 Model Overview

**Task:** Binary classification to predict patient readmission probability.

**Evaluation Metric:**  
- ROC-AUC  

### Features Used

- age  
- time_in_hospital  
- num_lab_procedures  
- num_medications  
- number_outpatient  
- number_emergency  
- number_inpatient  

**Model:**  
RandomForestClassifier (with class balancing and hyperparameter tuning)

---

## 🖥 Web Interface

A browser-based UI allows users to enter patient details and receive real-time readmission risk scores.

After starting the application, open:

```

[http://127.0.0.1:5000](http://127.0.0.1:5000)

````

---

## ⚙️ Run Locally (Without Docker)

```bash
pip install -r requirements.txt
python app.py
````

Then open:

```
http://127.0.0.1:5000
```

---

## 🐳 Run With Docker

Build the image:

```bash
docker build -t healthcare-ml .
```

Run the container:

```bash
docker run -p 5000:5000 healthcare-ml
```

Then open:

```
http://127.0.0.1:5000
```

---

## 🔌 REST API Usage

### Endpoint

```
POST /predict
```

### Sample Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "age":5,
  "time_in_hospital":3,
  "num_lab_procedures":40,
  "num_medications":10,
  "number_outpatient":1,
  "number_emergency":0,
  "number_inpatient":0
}'
```

### Sample Response

```json
{
  "readmission_risk": 0.46,
  "prediction": 0
}
```

---

## 📈 Monitoring

Prediction probabilities are logged to simulate basic model monitoring and drift detection.

---

## 📌 Key Engineering Learnings

* Built complete ML pipeline from preprocessing to deployment
* Improved ROC-AUC through feature engineering and class balancing
* Resolved training vs inference feature mismatch
* Implemented REST APIs with input validation
* Developed browser-based UI for ML inference
* Dockerized full-stack ML application
* Handled Python dependency conflicts and version alignment

---

## 👤 Author

**Dheeraj Saroha**
AI / Machine Learning Engineer

⚠️ Trained model file (`model.pkl`) is excluded from repository due to size limits.  
Run training script locally to generate model.