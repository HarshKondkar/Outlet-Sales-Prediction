import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import io

st.title('Shop Sales Prediction App')
st.subheader('This app predicts the outlet sales of a shop based on a few characteristics of the shop.')
train = pd.read_csv('Shop_Sales_Regression.csv')
st.write('The csv file you upload for prediction should be of the format:')
st.write(train.iloc[:, :-1].sample())

test = st.file_uploader('Upload your csv file below:')
if test is not None:
    test = pd.read_csv(test)
    st.write(test.head())
    train = train.dropna(axis = 0, how = 'any')
    test = test.dropna(axis = 0, how = 'any') # dropping NaN
    X = train.drop('Shop_Outlet_Sales', axis = 1)
    test_len = len(test)
    df = pd.concat([X, test], axis = 0)
    df = df.drop('Sl.No', axis = 1)
    df['Product_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'}, inplace = True)

    zhe_cols = ['Product_Fat_Content', 'Product_Type', 'Shop_Identifier', 'Shop_Size', 'Shop_Location_Type', 'Shop_Type']
    encoded = pd.get_dummies(df[zhe_cols], drop_first = True)
    df1 = pd.concat([df, encoded], axis = 1)
    df1.drop(zhe_cols, axis = 1, inplace = True)
    df1.drop(['Shop_Identifier_OUT018', 'Shop_Identifier_OUT035'], axis = 1, inplace = True)
    selected_cols = ['Product_MRP','Product_Visibility','Product_Weight','Shop_Establishment_Year',
                        'Product_Fat_Content_Regular','Product_Type_Snack Foods','Product_Type_Fruits and Vegetables',
                        'Shop_Identifier_OUT046','Product_Type_Household','Shop_Location_Type_Tier 3',
                        'Shop_Size_Small','Shop_Location_Type_Tier 2','Product_Type_Dairy','Product_Type_Frozen Foods',
                        'Shop_Identifier_OUT049']
    df1 = df1[selected_cols]

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    scaled_df = scaler.transform(df1)
    scaled = pd.DataFrame(data = scaled_df, columns = df1.columns)
    model = pickle.load(open('reg_model.pkl', 'rb'))
    predictions = model.predict(scaled)
    pred = pd.DataFrame(data = predictions, columns = ['Outlet_Sales'])
    final_df = pd.concat([scaled,pred], axis = 1)
    data = final_df.iloc[len(scaled)-test_len:, :]
    data = data.reset_index(drop = True)
    data = data.iloc[:,-1]
    st.write('### The predicted outlet sales for the above data are:')
    st.write(data)

    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="myfilename.xlsx">Download excel file</a>'
    st.markdown(linko, unsafe_allow_html=True)