import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

df = pd.read_csv('Notebook\data\creditCardFraud_28011964_120214.csv')


st.set_option('deprecation.showPyplotGlobalUse', False)
title1,title2,title3 = st.columns([1,10,1])
title2.title('Credit Card Fraud Detection')
st.sidebar.subheader('Navigation')
nav = st.sidebar.radio('',['Home','EDA','Prediction'])

if nav == 'Home':
    st.markdown("""
             ### **Made By - _Priykrit Varma_**
             ### **priykritv@gmail.com, priykrit21100@iiitnr.edu.in**
             ### **Contact no. - 9109562757**
             """)
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://static1.makeuseofimages.com/wordpress/wp-content/uploads/2015/12/credit-card-fraud.jpg");
    background-size: cover;
    
    }
    [data-testid="stHeadingWithActionElements"], 
    [data-testid="stHeadingWithActionElements"] * {
        color: black !important;
        font-weight: bold !important;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
elif nav == 'EDA':
    st.header('EDA')
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/58552pexels-fauxels-3183153-scaled.jpg");
    background-size: cover;
    }
    [data-testid="stHeadingWithActionElements"], 
    [data-testid="stHeadingWithActionElements"] * {
        color: black !important;
        font-weight: bold !important;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    if st.checkbox('Show Data'):
        st.write(df)
    
    if st.checkbox('Show Corelation Matrics'):
        corr = st.multiselect('Select desired Columns',list(df.columns),list(df.columns))
        plt.figure(figsize=(25,25))
        sns.heatmap(df[corr].corr(),annot=True)
        plt.title('Heat map of Selected columns')
        st.pyplot()
    
    
else:
    st.header('Enter values to predict price of diamond')
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://www.emailmeform.com/ssg/img/post/credit-card-fraud-types-hero.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    r11,r12,r13,r14,r15 =st.columns(5)
    LIMIT_BAL=r11.text_input('LIMIT_BAL')
    SEX=r12.selectbox('SEX',['1', '2'],index=None)
    EDUCATION=r13.selectbox('EDUCATION',['1', '2','3','4','5','6'],index=None)
    MARRIAGE=r14.selectbox('MARRIAGE',['0','1', '2','3'],index=None)
    AGE=r15.text_input('AGE')
    
    r21,r22,r23,r24,r25,r26 =st.columns(6)
    PAY_0=r21.text_input('PAY_0')
    PAY_2=r22.text_input('PAY_2')
    PAY_3=r23.text_input('PAY_3')
    PAY_4=r24.text_input('PAY_4')
    PAY_5=r25.text_input('PAY_5')
    PAY_6=r26.text_input('PAY_6')
    
    r31,r32,r33,r34,r35,r36 =st.columns(6)
    BILL_AMT1=r31.text_input('BILL_AMT1')
    BILL_AMT2=r32.text_input('BILL_AMT2')
    BILL_AMT3=r33.text_input('BILL_AMT3')
    BILL_AMT4=r34.text_input('BILL_AMT4')
    BILL_AMT5=r35.text_input('BILL_AMT5')
    BILL_AMT6=r36.text_input('BILL_AMT6')
    
    r41,r42,r43,r44,r45,r46 =st.columns(6)
    PAY_AMT1=r41.text_input('PAY_AMT1')
    PAY_AMT2=r42.text_input('PAY_AMT2')
    PAY_AMT3=r43.text_input('PAY_AMT3')
    PAY_AMT4=r44.text_input('PAY_AMT4')
    PAY_AMT5=r45.text_input('PAY_AMT5')
    PAY_AMT6=r46.text_input('PAY_AMT6')
    
    _,p,_ = st.columns(3)
    if p.button('Predict'):
        try:
            data = CustomData(
                float(LIMIT_BAL), 
                float(SEX), 
                float(EDUCATION), 
                float(MARRIAGE), 
                float(AGE), 
                float(PAY_0), 
                float(PAY_2), 
                float(PAY_3), 
                float(PAY_4), 
                float(PAY_5), 
                float(PAY_6), 
                float(BILL_AMT1), 
                float(BILL_AMT2), 
                float(BILL_AMT3), 
                float(BILL_AMT4), 
                float(BILL_AMT5), 
                float(BILL_AMT6), 
                float(PAY_AMT1), 
                float(PAY_AMT2), 
                float(PAY_AMT3), 
                float(PAY_AMT4), 
                float(PAY_AMT5), 
                float(PAY_AMT6)
            )
            
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            if int(pred)==1:
                st.error(f'Credit card holder will Default on their payment')
            else:
                st.success('Credit card holder Not Defaulted on their payment')
        except Exception as e:
            st.error('Some error occured')
            
