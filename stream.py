import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED 15-FEATURE MODEL
# ============================================================

model = joblib.load(
    "accident_severity_model_15_features.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Road Accident Severity Prediction")

st.write(
    "Enter the accident details below to predict the accident severity."
)

st.info("The prediction model uses 15 input features.")


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    # 1. Time
    Time = st.text_input(
        "1. Time",
        value="17:30"
    )


    # 2. Day of Week
    Day_of_week = st.selectbox(
        "2. Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


    # 3. Age Band of Driver
    Age_band_of_driver = st.text_input(
        "3. Age Band of Driver",
        value="18-30"
    )


    # 4. Sex of Driver
    Sex_of_driver = st.selectbox(
        "4. Sex of Driver",
        [
            "Male",
            "Female"
        ]
    )


    # 5. Driving Experience
    Driving_experience = st.text_input(
        "5. Driving Experience",
        value="1-2yr"
    )


    # 6. Type of Vehicle
    Type_of_vehicle = st.text_input(
        "6. Type of Vehicle",
        value="Automobile"
    )


    # 7. Area Accident Occurred
    Area_accident_occured = st.text_input(
        "7. Area Accident Occurred",
        value="Residential areas"
    )


    # 8. Types of Junction
    Types_of_Junction = st.text_input(
        "8. Types of Junction",
        value="Y Shape"
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    # 9. Road Surface Type
    Road_surface_type = st.text_input(
        "9. Road Surface Type",
        value="Asphalt roads"
    )


    # 10. Road Surface Conditions
    Road_surface_conditions = st.text_input(
        "10. Road Surface Conditions",
        value="Dry"
    )


    # 11. Light Conditions
    Light_conditions = st.text_input(
        "11. Light Conditions",
        value="Daylight"
    )


    # 12. Weather Conditions
    Weather_conditions = st.text_input(
        "12. Weather Conditions",
        value="Normal"
    )


    # 13. Type of Collision
    Type_of_collision = st.text_input(
        "13. Type of Collision",
        value="Vehicle with vehicle collision"
    )


    # 14. Number of Vehicles Involved
    Number_of_vehicles_involved = st.number_input(
        "14. Number of Vehicles Involved",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


    # 15. Number of Casualties
    Number_of_casualties = st.number_input(
        "15. Number of Casualties",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")


if st.button(
    "🚗 Predict Accident Severity",
    type="primary"
):

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    input_data = {

        "Time": Time,

        "Day_of_week": Day_of_week,

        "Age_band_of_driver": Age_band_of_driver,

        "Sex_of_driver": Sex_of_driver,

        "Driving_experience": Driving_experience,

        "Type_of_vehicle": Type_of_vehicle,

        "Area_accident_occured":
            Area_accident_occured,

        "Types_of_Junction":
            Types_of_Junction,

        "Road_surface_type":
            Road_surface_type,

        "Road_surface_conditions":
            Road_surface_conditions,

        "Light_conditions":
            Light_conditions,

        "Weather_conditions":
            Weather_conditions,

        "Type_of_collision":
            Type_of_collision,

        "Number_of_vehicles_involved":
            Number_of_vehicles_involved,

        "Number_of_casualties":
            Number_of_casualties
    }


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])


    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    feature_order = [
        "Time",
        "Day_of_week",
        "Age_band_of_driver",
        "Sex_of_driver",
        "Driving_experience",
        "Type_of_vehicle",
        "Area_accident_occured",
        "Types_of_Junction",
        "Road_surface_type",
        "Road_surface_conditions",
        "Light_conditions",
        "Weather_conditions",
        "Type_of_collision",
        "Number_of_vehicles_involved",
        "Number_of_casualties"
    ]

    input_df = input_df[feature_order]


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_df)[0]


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.success(
            f"### Predicted Accident Severity: {prediction}"
        )


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

