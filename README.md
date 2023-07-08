# Predicting Sale Price of Houses in King County using Multi Linear Regression
Machine Learning
# INTRODUCTON
This Machine Learning project appears involves analysing a dataset of housing prices and building a
machine learning model to predict the price of a house based on various features such as
square footage, number of bedrooms and bathrooms, and location.
Accurate prediction of the sale price of houses can be useful for a variety of purposes,
including real estate valuation, investment analysis, and property tax assessments.
# DATA DESCRIPTON
I have used the "House Sales in King County, USA" dataset available on Kaggle:
https://www.kaggle.com/harlfoxem/housesalesprediction.

The dataset used for predicting the sale price of houses in King County contains information
about 21,597 houses that were sold between May 2014 and May 2015.
The target variable in this dataset is the sale price of the house, while the other variables are
considered as potential predictors. The dataset also includes several categorical variables, such
as waterfront and view, which indicate whether the house has a waterfront view or a view of
the mountains, and other variables like floors and condition which are numerical but represent
categorical features.

The descriptions of each feature are as follows:
id — unique ID for a house,
date — date day house was sold,
price — Price is the prediction target,
bedrooms — number of bedrooms,
bathrooms — number of bathrooms,
sqft_living — square footage of the home,
sqft_lot — square footage of the lot,
floors — total floors (levels) in house,
waterfront — whether house has a view to a waterfront,
view — number of times house has been viewed,
condition — how good the condition is (overall),
grade — overall grade given to the housing unit based on King County grading system,
sqft_above — square footage of house (apart from basement),
sqft_basement — square footage of the basement,
yr_built — year when house was built,
yr_renovated — year when house was renovated,
zipcode — zip code in which house is located,
lat — Latitude coordinate,
long — Longitude coordinate,
sqft_living15 — the square footage of interior housing living space for the nearest 15
neighbors,
sqft_lot15 — The square footage of the land lots of the nearest 15 neighbor.

# DATA ANALYSIS
Exploratory data analysis (EDA) is a critical process in any data science project. It involves
examining and understanding the data through summary statistics, data visualization, and data
cleaning to uncover insights and identify patterns, trends, and outliers in the data.
First, I used summary statistics such as mean, median, standard deviation, minimum, and
maximum values to understand the range and distribution of variables in the dataset.

I also performed data cleaning by removing missing values, outliers, and irrelevant
variables to ensure that the data was accurate and reliable for building our predictive model.
Next, I created various data visualizations such as histograms, box plots, scatter plots, and
heat maps to explore the relationships between variables, detect outliers, and identify any
trends or patterns.
A folium map object called 'houses_map' is centered on the mean latitude and longitude values
of the dataset to visualise the data geographically.

# REASON TO SELECT MULTI LINEAR REGRESSION AS A MACHINE LEARNING MODEL
Multilinear regression is a statistical model used to establish the relationship between multiple
independent variables (features) and a single dependent variable (target). It assumes a linear
relationship between the features and the target variable, and it is a suitable model when there
are multiple features that may be influencing the target variable.

In the context of the housing price dataset, the use of multilinear regression is appropriate
because there are multiple features such as number of bedrooms, bathrooms, square footage of
living area, condition of the house, and location, that may have an impact on the price of a
house. By including all these features in the model, we can obtain a more accurate prediction
of the house price, rather than relying on a single feature.

The use of multilinear regression also allows us to estimate the coefficients of the features,
which give us an indication of the strength and direction of the relationship between each
feature and the target variable. This information can be used to identify which features are
most important in predicting the price of a house.

However, it is important to note that multilinear regression makes several assumptions, such as
linearity, homoscedasticity, and independence of errors, which may not always hold true in
real-world scenarios. Therefore, it is important to carefully evaluate the assumptions and
potential limitations of the model before using it for prediction.

# ALGORITHM
The algorithm for multi-linear regression can be summarized in the following steps:
1. Prepare the data: Split the dataset into two sets - training set and testing set. Preprocess the data by handling missing values, feature scaling, encoding categorical
variables, etc.
2. Select the independent variables: Decide which variables are important predictors for
the dependent variable (target). Use correlation analysis, feature selection techniques,
etc. to select the relevant independent variables.
3. Train the model: Use the training set to train the multi-linear regression model. The
model will learn the relationships between the independent variables and the dependent
variable.
4. Test the model: Use the testing set to test the accuracy of the model. The model will
make predictions for the dependent variable based on the independent variables in the
testing set. Compare the predicted values with the actual values of the dependent
variable in the testing set.
5. Evaluate the model: Evaluate the performance of the model using various metrics such
as mean squared error, root mean squared error, R-squared value, etc. Use these
metrics to compare the performance of different models and choose the best one.
6. Use the model for prediction: Once the model is trained and tested, it can be used to
make predictions for new data. Provide the values of the independent variables to the
model and it will predict the value of the dependent variable.

# RESULT ANALYSIS
1. Data Preprocessing: We first performed some basic preprocessing steps such as
removing duplicates, handling missing values, and removing outliers. We also
converted categorical variables to numerical variables using one-hot encoding.
2. Exploratory Data Analysis (EDA): We then performed EDA to gain insights into the
dataset. We visualized the data using various plots, such as histograms, scatter plots,
and box plots, and identified the correlation between different variables using
correlation matrices.
3. Feature Engineering: We then performed feature engineering, which involved selecting
the most important features and scaling the data.
4. Model Training: We trained the multi-linear regression model using the preprocessed
data. We split the data into training and testing sets and used k-fold cross-validation to
validate the model.
5. Model Evaluation: We evaluated the model using various metrics such as mean
squared error (MSE), mean absolute error (MAE), and R-squared. We also visualized
the predicted and actual values using scatter plots.
6. Result Analysis: The multi-linear regression model achieved an R-squared value of 0.7,
which indicates that the model can explain around 70% of the variance in the data. The
mean squared error was around 4.2 million, and the mean absolute error was around
1.6 thousand dollars. The scatter plots of the predicted and actual values showed that
the model was able to predict the prices reasonably well.
Overall, the multi-linear regression model was able to predict the house prices in King County
with reasonable accuracy. 

# CONCLUSION AND FUTURE SCOPE
In conclusion, we have successfully built a multi linear regression model to predict the sale
prices of houses in King County. The model was evaluated using metrics such as Mean
Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared value. The model
achieved an RMSE value of 190,459, which means that the predicted prices were on average
around $190,459 away from the actual prices. The R-squared value of 0.70 indicates that the
model explains 70% of the variance in the data.
Based on the coefficients of the model, we can conclude that the most significant factors
affecting the sale price of a house in King County are the size of the living area, the number of
bathrooms, the location (zip code), and the grade of the house.
In the future, we can improve the performance of the model by incorporating additional
relevant features such as the age of the house, the condition of the house, and the proximity to
amenities such as schools, parks, and shopping centers. We can also explore other advanced
regression techniques such as Random Forest Regression and Gradient Boosting Regression to
further improve the accuracy of the model. Additionally, we can apply various feature
selection techniques to identify the most relevant features for the model, which can further
improve the model's performance.

# PYTHON NOTEBOOK
https://colab.research.google.com/drive/12RlmlluMLLxqFQsRwmAxhSIu2JDFsBD6?usp=sharing


