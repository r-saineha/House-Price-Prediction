import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("house_price_model.pkl")

# ----------------------------
# Load Dataset (used for graphs)
# ----------------------------
df = pd.read_csv("data (1).csv")

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
#left_col, right_col = st.columns([1, 1.5])
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h2 style='font-size:32px;text-align:center;'>🏠 House Price Prediction</h2>",
    unsafe_allow_html=True
)
st.write("Enter the house details below")
left_col, right_col = st.columns([1, 1.3])
# ----------------------------
# User Inputs
# ----------------------------
with left_col:
    st.subheader("House Details")
    bedrooms = st.number_input( "Bedrooms", min_value=0, max_value=20, value=3, width=500 )
    bathrooms = st.number_input( "Bathrooms", min_value=0.0, max_value=10.0, value=2.0, width=500 )
    sqft_living = st.number_input( "Living Area (sqft)", min_value=0, value=1500, width=500 )
    sqft_lot = st.number_input( "Lot Area (sqft)", min_value=0, value=5000, width=500 )
    floors = st.number_input( "Floors", min_value=1.0, max_value=5.0, value=1.0, width=500 )
    waterfront = st.selectbox( "Waterfront", [0, 1], width=500 )
    view = st.slider( "View Rating", 0, 4, 0, width=500 )
    condition = st.slider( "Condition", 1, 5, 3, width=500 )
    sqft_above = st.number_input( "Sqft Above Ground", min_value=0, value=1200, width=500 )
    sqft_basement = st.number_input( "Basement Sqft", min_value=0, value=300, width=500 )
    yr_built = st.number_input( "Year Built", min_value=1800, max_value=datetime.today().year, value=2000, width=500 )
    yr_renovated = st.number_input( "Year Renovated", min_value=0, max_value=datetime.today().year, value=0, width=500 )
    street = st.selectbox( "Street", sorted(df["street"].unique()), width=500 )
    city = st.selectbox( "City", sorted(df["city"].unique()), width=500 )
    statezip = st.selectbox( "State Zip", sorted(df["statezip"].unique()), width=500 )
    country = st.selectbox( "Country", sorted(df["country"].unique()), width=500 )
    date = st.date_input( "Sale Date", datetime.today(), width=500 )

# ----------------------------
# Convert Date
# ----------------------------

year = date.year
month = date.month
day = date.day
if "input_data" not in st.session_state:
    st.session_state.input_data = None
# ----------------------------
# Prediction
# ----------------------------
with right_col:
    if st.button("Predict Price"):

        input_data = pd.DataFrame({

            "bedrooms":[bedrooms],
            "bathrooms":[bathrooms],
            "sqft_living":[sqft_living],
            "sqft_lot":[sqft_lot],
            "floors":[floors],
            "waterfront":[waterfront],
            "view":[view],
            "condition":[condition],
            "sqft_above":[sqft_above],
            "sqft_basement":[sqft_basement],
            "yr_built":[yr_built],
            "yr_renovated":[yr_renovated],
            "street":[street],
            "city":[city],
            "statezip":[statezip],
            "country":[country],
            "year":[year],
            "month":[month],
            "day":[day]
        })
    

        prediction = model.predict(input_data)[0]
    
        st.success(f"Estimated House Price: ${prediction:,.2f}")

       
        fig, ax = plt.subplots(figsize=(5,3))

        sns.histplot(df["price"], bins=30, kde=True, color="skyblue", ax=ax)

        ax.axvline(
            prediction,
            color="red",
            linewidth=3,
            label="Predicted Price"
    )

        ax.legend()

        ax.set_title("Price Distribution")

        st.pyplot(fig)

    # -------------------------------
    # Living Area vs Price
    # -------------------------------
        fig, ax = plt.subplots(figsize=(5,3))

        sns.scatterplot(
            x="sqft_living",
            y="price",
            data=df,
            alpha=0.6,
            ax=ax
        )

        ax.scatter(
            sqft_living,
            prediction,
            color="red",
            s=150,
            label="Predicted House"
    )

        ax.legend()

        ax.set_title("Living Area vs Price")

        st.pyplot(fig)

    # -------------------------------
    # Bedrooms vs Price
    # -------------------------------
        fig, ax = plt.subplots(figsize=(5,3))

        sns.boxplot(
        x="bedrooms",
        y="price",
        data=df,
        ax=ax
        )

        ax.scatter(
        bedrooms,
        prediction,
        color="red",
        s=120,
        label="Prediction"
    )

        ax.legend()

        st.pyplot(fig)

   
