import streamlit as st
import pandas as pd
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


def user_input_features() -> pd.DataFrame:
    building = st.sidebar.selectbox('Type de bien', options=building_options.keys())
    quality = st.sidebar.slider('Qualité du bien', int(X["Overall Qual"].min()), int(X["Overall Qual"].max()))
    age = st.sidebar.selectbox('Ancienneté du bien', options=age_options.keys())
    lot_area = st.sidebar.slider('Surface totale', int(X["Lot Area"].min()), int(X["Lot Area"].max()), int(X["Lot Area"].mean()))
    gr_liv_area = st.sidebar.slider('Surface au sol', int(X["Gr Liv Area"].min()), int(X["Gr Liv Area"].max()), int(X["Gr Liv Area"].mean()))
    lot_frontage = st.sidebar.slider('Taille de la façade', int(X["Lot Frontage"].min()), int(X["Lot Frontage"].max()), int(X["Lot Frontage"].mean())),
    bedrooms = st.sidebar.slider('Nombre de chambres', 1, int(X["Bedroom AbvGr"].max()), step=1)
    kitchens = st.sidebar.slider('Nombre de cuisines', 1, int(X["Kitchen AbvGr"].max()), step=1)
    bathrooms = st.sidebar.slider('Nombre de salles de bains', 1, int(X["Bathrooms"].max()), step=1)

    data = {
        'AgeBins': age_options[age],
        'Bldg Type': building_options[building],
        'Gr Liv Area': gr_liv_area,
        'Lot Frontage': lot_frontage,
        'Lot Area': lot_area,
        'Bedroom AbvGr': bedrooms,
        'Kitchen AbvGr': kitchens,
        'Bathrooms': bathrooms,
        'Overall Qual': quality,
    }
    return pd.DataFrame(data, index=[0])


def user_input_extra_features() -> pd.DataFrame:
    with st.expander('Plus de critères'):
        floor = st.checkbox('Avec étage?')
        basement = st.checkbox('Avec sous-sol?')
        garage = st.checkbox('Avec garage?')
        fireplace = st.checkbox('Avec cheminée?')

    data = {
        "Has2ndFloor": 1 if floor else 0,
        "HasBasement": 1 if basement else 0,
        "HasGarage": 1 if garage else 0,
        "has_fireplace": 1 if fireplace else 0,
    }
    return pd.DataFrame(data, index=[0])


df_1 = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Précisez vos critères')
df_2 = user_input_extra_features()
st.write('---')

df = pd.concat([df_1, df_2], axis=1)
st.write(df)
# Apply Model to Make Prediction
loaded_model = pyc.load_model('./upgrade/fine_tuned_lgbm')
prediction = loaded_model.predict(df)

formated_prediction = '${:,}'.format(int(prediction))
st.header('Prediction du prix de vente')
st.write(formated_prediction)
st.write('---')
