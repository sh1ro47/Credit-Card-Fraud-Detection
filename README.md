## Credit Card Fraud Detection
> Comprehensive Credit Card Fraud Detection project adhering to _**Industry Standards**_.
>
> ![cc1](https://github.com/Priykrit/Credit-Card-Fraud-Detection/assets/98400044/d296249c-a718-4703-ab4e-ff886c716b0d)

> Developed _**Pipelines**_ for training and prediction, including **_automatic model selection_** based on precision. Implemented robust logging and exception-handling mechanisms.
>
> Selected _Naive Bayes_ as the best model with a **_precision_** of **_0.62_** for predicting _customer default on the next payment_.
>
> Utilized Streamlit to create an interactive interface with dedicated sections for:
>
> > Exploratory Data Analysis (EDA) included visualizations such as a correlation matrix heatmap.
> >
> > A dedicated page predicts whether the customer will default on the next payment.
> >
> > > ![cc2](https://github.com/Priykrit/Credit-Card-Fraud-Detection/assets/98400044/5d6303c9-e774-4d3a-8539-5fe6e9b7a295)
> > >
> > > ![cc3](https://github.com/Priykrit/Credit-Card-Fraud-Detection/assets/98400044/8e88175a-40b6-43fc-8e20-12934c3ac0ca)


### To Run project
> Create venv with _**python==3.8.0**_
> 
> Install all requirements with _**pip install -r requirements.txt**_
> 
> Run _**python src/pipelines/training_pipeline.py**_ in terminal from root dir to create preprocessor and model file
> 
> Run _**streamlit run application.py**_ in terminal from root dir to run the web app
