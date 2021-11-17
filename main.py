import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.write("""
# Prédiction du prix de vente des biens immobiliers à Ames (Iowa USA)
""")
st.write('---')


X = pd.read_csv("clean_X.csv")


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Quels sont vos critères?')


def user_input_features():
    Age = st.sidebar.slider('Ancienneté du bien', int(X.Age.min()), int(X.Age.max()), int(X.Age.mean()))
    LotArea = st.sidebar.slider('Surface totale', int(X.LotArea.min()), int(X.LotArea.max()), int(X.LotArea.mean()))
    GrLivArea = st.sidebar.slider('Surface au sol', int(X.GrLivArea.min()), int(X.GrLivArea.max()), int(X.GrLivArea.mean()))
    LotFrontage = st.sidebar.slider('Taille de la façade', int(X.LotFrontage.min()), int(X.LotFrontage.max()), int(X.LotFrontage.mean()))
    GarageArea = st.sidebar.slider('Taille du garage', int(X.GarageArea.min()), int(X.GarageArea.max()), int(X.GarageArea.mean()))
    Fence = st.sidebar.select_slider('Présence de barrières', options=[False, True])
    Pool = st.sidebar.select_slider('Piscine souhaitée?', options=[False, True])

    data = {'Age': Age,
            'GrLivArea': GrLivArea,
            'LotFrontage': LotFrontage,
            'LotArea': LotArea,
            'GarageArea': GarageArea,
            'Fence': Fence,
            'Pool' : Pool
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Précisez vos critères')
st.write(df)
st.write('---')


# Apply Model to Make Prediction
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
prediction = loaded_model.predict(df)

formated_prediction = '${:,}'.format(int(prediction))
st.header('Prediction du prix de vente')
st.write(formated_prediction)
st.write('---')
