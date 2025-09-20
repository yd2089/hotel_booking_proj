# Hotel Booking Analysis & Prediction Project

## 🎯 Objective
Analyze hotel booking data to identify key patterns, predict cancellations, and segment customers for personalized marketing strategies.

## 📊 Project Overview
This project performs comprehensive analysis on hotel booking data using machine learning techniques to:
- **Predict booking cancellations** using multiple ML algorithms
- **Segment customers** through clustering analysis
- **Identify patterns** in booking behavior and seasonal trends
- **Provide actionable insights** for hotel revenue optimization

## 📁 Project Structure
```
├── EDA_Clustering.ipynb          # Exploratory Data Analysis & Customer Segmentation
├── model.ipynb                   # Machine Learning Models for Cancellation Prediction
├── hotel_bookings.csv            # Dataset (119,390 hotel booking records)
├── ML_Report.pdf                 # Detailed project report
├── ML_finalpresentation.pdf      # Project presentation
└── README.md                     # This file
```

## 🔍 Dataset Information
- **Source**: [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) from Kaggle
- **Description**: Hotel booking data from two hotels (City Hotel & Resort Hotel)
- **Size**: 119,390 records with 32 features
- **Time Period**: July 2015 - August 2017
- **Key Features**: 
  - Booking details (lead time, arrival date, length of stay)
  - Customer information (country, market segment, customer type)
  - Hotel characteristics (hotel type, room type, meal plan)
  - Booking status (cancellation, deposit type)

## 🛠️ Methodology

### 1. Exploratory Data Analysis (`EDA_Clustering.ipynb`)
- **Data preprocessing** and missing value handling
- **Statistical analysis** of booking patterns
- **Visualization** of seasonal trends and customer behavior
- **Customer segmentation** using K-Means clustering
- **Feature correlation** analysis

### 2. Machine Learning Models (`model.ipynb`)
- **Data preprocessing** with feature engineering
- **Dimensionality reduction** using PCA
- **Multiple ML algorithms**:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier
- **Model evaluation** with cross-validation
- **Performance comparison** across different algorithms

## 🎯 Key Findings

### Customer Segmentation
- **3 distinct customer clusters** identified based on lead time and average daily rate (ADR)
- **Cluster characteristics**:
  - High-value, early bookers
  - Price-sensitive, last-minute bookers
  - Moderate-value, regular bookers

### Cancellation Prediction
- **Best performing model**: Random Forest with optimized hyperparameters
- **Key predictors**: Lead time, deposit type, customer type, market segment
- **Model performance**: Achieved high accuracy in predicting booking cancellations

### Business Insights
- **Seasonal patterns** in booking and cancellation rates
- **Customer behavior** differences between hotel types
- **Revenue optimization** opportunities through targeted marketing

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running the Analysis
1. **EDA and Clustering**: Open `EDA_Clustering.ipynb` and run all cells
2. **ML Models**: Open `model.ipynb` and run all cells
3. **View Results**: Check the generated visualizations and model performance metrics

## 📈 Results & Performance
- **Clustering**: Successfully identified 3 customer segments using K-Means
- **Classification**: Multiple models tested with cross-validation
- **Feature Importance**: Lead time and deposit type are key predictors
- **Model Optimization**: Hyperparameter tuning to reduce overfitting

## 📋 Files Description
- **`EDA_Clustering.ipynb`**: Complete exploratory data analysis and customer segmentation
- **`model.ipynb`**: Machine learning pipeline for cancellation prediction
- **`hotel_bookings.csv`**: Original dataset
- **`ML_Report.pdf`**: Comprehensive project report
- **`ML_finalpresentation.pdf`**: Project presentation slides

## 🎓 Academic Context
This project was developed as part of the Machine Learning I course at the University of Chicago, demonstrating practical application of machine learning techniques to real-world business problems.

## 📞 Contact
For questions about this project, please refer to the detailed analysis in the Jupyter notebooks or the comprehensive report in `ML_Report.pdf`.

---
*Last updated: September 2024*
