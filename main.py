import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pycaret.regression as pyc

st.set_page_config(page_title='House prices', layout='centered')

st.write("""
# Prédiction du prix de vente des biens immobiliers à Ames (Iowa USA)
""")
st.write('---')


X = pd.read_csv("./upgrade/data/02/cleaned_ames_full.csv", index_col=0)

age_options = {
    "0 à 5 ans": 0,
    "5 à 20 ans": 1,
    "20 à 50 ans": 2,
    "50+ ans": 3
}

building_options = {
    "Maison individuelle": "1Fam",
    "Maison mitoyenne": "2FmCon",
    "Duplex": "Duplx",
    "Maison de ville (extérieur)": "TwnhsE",
    "Maison de ville (intérieur)": "TwnhsI",
}


# Sidebar
# Header of Specify Input Parameters
st.sidebar.write('Quels sont vos critères?')
def user_input_features():
    Building = st.sidebar.selectbox('Type de bien', options=building_options.keys())
    Quality =  st.sidebar.slider('Qualité du bien', int(X["Overall Qual"].min()), int(X["Overall Qual"].max()))
    Age = st.sidebar.selectbox('Ancienneté du bien', options=age_options.keys())
    LotArea = st.sidebar.slider('Surface totale', int(X["Lot Area"].min()), int(X["Lot Area"].max()), int(X["Lot Area"].mean()))
    GrLivArea = st.sidebar.slider('Surface au sol', int(X["Gr Liv Area"].min()), int(X["Gr Liv Area"].max()), int(X["Gr Liv Area"].mean()))
    LotFrontage = st.sidebar.slider('Taille de la façade', int(X["Lot Frontage"].min()), int(X["Lot Frontage"].max()), int(X["Lot Frontage"].mean())),
    Bedrooms = st.sidebar.slider('Nombre de chambres', int(X["Bedroom AbvGr"].min()), int(X["Bedroom AbvGr"].max()), step=1)
    Kitchens = st.sidebar.slider('Nombre de cuisines', int(X["Kitchen AbvGr"].min()), int(X["Kitchen AbvGr"].max()), step=1)
    Bathrooms = st.sidebar.slider('Nombre de salles de bains', int(X["Bathrooms"].min()), int(X["Bathrooms"].max()), step=1)

    data = {
        'AgeBins': age_options[Age],
        'Bldg Type': building_options[Building],
        'Gr Liv Area': GrLivArea,
        'Lot Frontage': LotFrontage,
        'Lot Area': LotArea,
        'Bedroom AbvGr': Bedrooms,
        'Kitchen AbvGr': Kitchens,
        'Bathrooms': Bathrooms,
        'Overall Qual': Quality,
    }
    return pd.DataFrame(data, index=[0])

def user_input_extra_features():
    with st.expander('Plus de critères'):
        Floor = st.checkbox('Avec étage?')
        Basement = st.checkbox('Avec sous-sol?')
        Garage = st.checkbox('Avec garage?')
        Fireplace = st.checkbox('Avec cheminée?')

    data = {
        "Has2ndFloor": Floor,
        "HasBasement": Basement,
        "HasGarage": Garage,
        "has_fireplace": Fireplace,
    }
    return pd.DataFrame(data, index=[0])

df_1 = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Précisez vos critères')
df_2 = user_input_extra_features()
st.write('---')

df = pd.concat([df_1, df_2], axis=1)

# Apply Model to Make Prediction
loaded_model = pyc.load_model('./upgrade/fine_tuned_lgbm')
prediction = loaded_model.predict(df)

formated_prediction = '${:,}'.format(int(prediction))
st.header('Prediction du prix de vente')
st.write(formated_prediction)
st.write('---')
