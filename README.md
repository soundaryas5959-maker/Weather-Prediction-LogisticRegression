## 🤖 Model Details - Logistic Regression

This project uses **Multinomial Logistic Regression** for multiclass weather classification.

**Why Logistic Regression?**
- Best for categorical output with multiple classes
- Gives probability scores for each weather class
- Fast training & prediction time
- Works well with linearly separable data

**Model Pipeline:**
1. **Input**: Temperature, Humidity, Apparent Temperature, Pressure
2. **Algorithm**: Softmax function used to predict 5 classes
3. **Output**: Clear, Rainy, Partly Cloudy, Mostly Cloudy, Foggy

**Training:**
- Dataset split: 80% train, 20% test
- Feature scaling applied using StandardScaler
- Model saved as `model (4).pkl` using Joblib
