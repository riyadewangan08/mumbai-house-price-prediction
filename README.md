# mumbai-house-price-prediction
# 🏠 Mumbai House Price Prediction using Machine Learning

This project predicts house prices in Mumbai using advanced regression models in Python. It includes complete steps from data preprocessing, visualization, model building to prediction, designed to reflect real-world data analyst practices.

---

## 📂 Dataset Overview

- **Format:** CSV  
- **Rows:** 1000+  
- **Columns:** 9+  
- **Features:** Location, Area (sqft), BHK, Bathroom, Furnishing, Parking, Status, Transaction Type, Price  

> ✅ Missing values handled using imputation & dropping strategies  
> 📍 Data cleaned and encoded for ML models  

---

## 🔍 Exploratory Data Analysis (EDA)

### ▶️ Missing Values Heatmap  
Shows which columns contain null values.  
📌 *Used to decide what to drop/fill.*

📷 ![Missing Values Heatmap](https://github.com/user-attachments/assets/15b62315-081f-4c3d-bb17-d49c73076e9a)

### ▶️ Price Distribution  
Displays how house prices are spread across the dataset.

📷 ![Price Distribution](https://github.com/user-attachments/assets/6b7a84c6-3293-49b9-95a5-ba877e23f8bc)

### ▶️ Correlation Heatmap  
Shows correlation between numeric features to understand dependencies.

📷 ![Correlation Heatmap](https://github.com/user-attachments/assets/5da883c7-9c14-4fa9-b3c5-f26f305f2995)

---

## 📊 Feature Engineering & Encoding

- Used **`get_dummies()`** for categorical data  
- Filled selected null values using `mean()` or `mode()`  
- Dropped outliers and high-missing-value columns
- Encoded final dataset for model training

---

## 🧠 Models Applied

1. **Linear Regression**
2. **Lasso Regression**
3. **Random Forest Regression**

Each model was trained and evaluated on the same split data.

---

## 📈 Model Evaluation

Metrics used:
- RMSE (Root Mean Squared Error)
- R² Score (Goodness of fit)

| Model             | R² Score | RMSE       |
|------------------|----------|------------|
| Linear Regression| 0.82     | ₹19.18 Lakh|
| Lasso Regression | 0.82     | ₹19.18 Lakh|
| Random Forest    | 0.77     | ₹21.89 Lakh|

📷 ![Prediction Comparison Graph](images/final_output.png)

---

## 🌟 Feature Importance (Random Forest)

Helps us know which features influenced price prediction the most.

📷 ![Feature Importance](https://github.com/user-attachments/assets/eb8b280f-074c-4e29-831b-a9c2235ab240)

---

## 🧪 Residuals Plot

Checks how much predicted price deviated from actual price.

📷 ![Residuals Distribution](images/residual_plot.png)

---

## 📦 Prediction Example

We predicted house price using a random sample from the test set.

```python
new_house = X_test.iloc[1].values.reshape(1, -1)
predicted_price = rf.predict(new_house)[0]
print(f"Predicted price: ₹{round(predicted_price)}")


## 📌 Business Insights
📍 Location, Area, and Number of Bedrooms play the most crucial role in predicting prices.

🧠 Random Forest provides feature importance, even if its accuracy is slightly lower.

🔁 Linear and Lasso performed equally with higher R² values.

📈 Residuals indicate the model fits well (errors are normally distributed).

🔍 EDA helped in detecting outliers and missing trends

📃 Conclusion
✅ This project reflects a full ML pipeline for regression analysis.

✅ Performed complete data cleaning, visualization, modeling, and evaluation.

📌 Can be extended with more features like nearby amenities, age of property, etc.

📦 Model saved using joblib for deployment.

# 👩‍💻 Tools Used
Python, Jupyter Notebook

Pandas, NumPy, Seaborn, Matplotlib

Scikit-learn

Joblib (for model export)

📬 Contact
Name: Riya
Email: riyadew77@gmail.com


