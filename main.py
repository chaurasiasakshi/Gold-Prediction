import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

st.title("Gold Data Visualization")

# Read csv file
data = pd.read_csv("gld_price_data.csv")



if st.sidebar.button('Load Description'):
    st.write(data.describe())
    
    
img = Image.open('gold.jpeg')
st.image(img,width=50,use_column_width=True)

if st.sidebar.button('Load Dataset'):
    st.write(data)
    
    

# Split Data Set

X = data.drop(['Date', 'GLD'], axis=1)
Y = data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)

# Training the Model
regressor.fit(X_train, Y_train)
pred_train = regressor.predict(X_train)  # Predictions on the training set
pred_test = regressor.predict(X_test)    # Predictions on the test set


if st.sidebar.button('Load Chart'):
    fig, (ax1) = plt.subplots( figsize=(10, 6))

 

    # Plotting Distribution Plot for Test Data
    sns.histplot(Y_test - pred_test, kde=True, ax=ax1)
    ax1.set_title("Test Set Residual Distribution")
    ax1.set_xlabel('Residual Values')
    ax1.set_ylabel('Frequency')

    st.pyplot(fig)
