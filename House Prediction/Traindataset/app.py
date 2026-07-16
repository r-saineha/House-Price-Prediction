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
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below")

# ----------------------------
# User Inputs
# ----------------------------

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=20,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    value=1500
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=5.0,
    value=1.0
)

waterfront = st.selectbox(
    "Waterfront",
    [0,1]
)

view = st.slider(
    "View Rating",
    0,
    4,
    0
)

condition = st.slider(
    "Condition",
    1,
    5,
    3
)

sqft_above = st.number_input(
    "Sqft Above Ground",
    value=1200
)

sqft_basement = st.number_input(
    "Basement Sqft",
    value=300
)

yr_built = st.number_input(
    "Year Built",
    value=2000
)

yr_renovated = st.number_input(
    "Year Renovated",
    value=0
)

street = st.selectbox("Street", sorted(df["street"].unique()))
city = st.selectbox("City", sorted(df["city"].unique()))
statezip = st.selectbox("State Zip", sorted(df["statezip"].unique()))
country = st.selectbox("Country", sorted(df["country"].unique()))

date = st.date_input(
    "Sale Date",
    datetime.today()
)

# ----------------------------
# Convert Date
# ----------------------------

year = date.year
month = date.month
day = date.day

# ----------------------------
# Prediction
# ----------------------------

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
    st.subheader("Input Data")
    st.write(input_data)

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")

    st.markdown("---")
    st.subheader("Prediction Analysis")
    st.write("Prediction:", prediction)
    # -------------------------------
    # Price Distribution
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8,5))

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
    fig, ax = plt.subplots(figsize=(8,5))

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
    fig, ax = plt.subplots(figsize=(8,5))

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

    # -------------------------------
    # Bathrooms vs Price
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        x="bathrooms",
        y="price",
        data=df,
        ax=ax
    )

    ax.scatter(
        bathrooms,
        prediction,
        color="red",
        s=120
    )

    st.pyplot(fig)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10,8))

    numeric_df = df.select_dtypes(include=["int64","float64"])

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

   