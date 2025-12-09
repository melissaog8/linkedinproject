import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# Custom App Styling 
st.markdown(
    """
    <style>
        /* Background */
        .stApp {
            background-color: #E8F3FF;
        }

        /* Body text */
        html, body, [class*="css"]  {
            color: #1D2226;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #0A66C2;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load trained model
with open("linkedin_model.pkl", "rb") as f:
    model = pickle.load(f)

# LinkedIn logo
st.markdown(
    """
    <div style="position: absolute; top: 20px; right: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="70">
    </div>
    """,
    unsafe_allow_html=True
)

# Header 
st.title("Who’s on LinkedIn?")

# Create tabs for the app to have the predictor and the visuals
tab1, tab2 = st.tabs(["🔍 Predictor Tool", "📊 User Insights Dashboard"])

# Tab 1: Linkedin Predictor Tool
with tab1:

    st.header("🔍 LinkedIn Predictor Tool")
    st.subheader("Enter your details to see how the model predicts you as a LinkedIn user.")

    # Consent message
    consent = st.checkbox("I agree to provide my information for learning purposes only.")

    if consent:
        st.divider()

        # User input
        age = st.slider("Age:", 1, 98, 25)

        female = st.selectbox("What is your gender?", ["Female", "Male"])
        female_val = 1 if female == "Female" else 0  # 1 = female, 0 = male

        education_options = {
            "Less than high school": 1,
            "High school incomplete": 2,
            "High school graduate": 3,
            "Some college": 4,
            "College graduate": 5,
            "Post-college": 6,
            "Professional degree": 7,
            "Doctorate": 8
        }
        education_label = st.selectbox(
            "What is your level of education?",
            list(education_options.keys())
        )
        education = education_options[education_label]

        income_options = {
            "Less than $10,000": 1,
            "$10,000–$19,999": 2,
            "$20,000–$29,999": 3,
            "$30,000–$39,999": 4,
            "$40,000–$49,999": 5,
            "$50,000–$74,999": 6,
            "$75,000–$99,999": 7,
            "$100,000–$149,999": 8,
            "$150,000 or more": 9
        }
        income_label = st.selectbox("What is your income range?", list(income_options.keys()))
        income = income_options[income_label]

        parent = st.radio("Do you have any children?", ["No", "Yes"])
        parent_val = 1 if parent == "Yes" else 0

        married = st.radio("Are you currently married?", ["No", "Yes"])
        married_val = 1 if married == "Yes" else 0

        # Model input
        input_data = pd.DataFrame({
            "income": [income],
            "education": [education],
            "parent": [parent_val],
            "married": [married_val],
            "female": [female_val],
            "age": [age]
        })

        # Results 
        if st.button("Am I a LinkedIn user?"):
            prob = model.predict_proba(input_data)[0][1]
            prediction = model.predict(input_data)[0]

            st.write(f"**Your probability of being a LinkedIn user:** {prob:.3f}")

            if prediction == 1:
                st.warning("You are predicted to be a LinkedIn user! Professional mode: ON 💼")
            else:
                st.warning("You are predicted *not* to be a LinkedIn user. Offline mode: ON 📵")

    else:
        st.info("⚠️ Please agree to provide your information before using the predictor tool.")

# Tab 2: User Insights Dashboard
with tab2:
    st.header("📊 User Insights Dashboard")
    st.write("Here are some interesting insights from the dataset that was used to build the prediction model.")

    # Loading the raw dataset to build the visualizations
    li_df = pd.read_csv("social_media_usage.csv")

    # Create new dataframe with relevant columns
    li_df = li_df[["web1h", "income", "educ2", "par", "marital", "gender", "age"]].copy()

    # Create sm_li using clean_sm
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    li_df["sm_li"] = clean_sm(li_df["web1h"])

    # Rename features
    li_df = li_df.rename(columns={
        "educ2": "education",
        "par": "parent",
        "marital": "married",
        "gender": "female"
    })

    # Feature transformation
    li_df["income"] = np.where(li_df["income"].between(1, 9), li_df["income"], np.nan)
    li_df["education"] = np.where(li_df["education"].between(1, 8), li_df["education"], np.nan)
    li_df["parent"] = np.where(
        li_df["parent"] == 1, 1,
        np.where(li_df["parent"] == 2, 0, np.nan)
    )
    li_df["married"] = np.where(li_df["married"] == 1, 1, 0)
    li_df["female"] = np.where(li_df["female"] == 2, 1, 0)
    li_df["age"] = np.where(li_df["age"] <= 97, li_df["age"], np.nan)

    # Drop missing values
    li_df = li_df.dropna()

    # No longer needed
    li_df = li_df.drop(columns=["web1h"])

    # Rename sm_li to linkedin for readability
    li_df = li_df.rename(columns={"sm_li": "linkedin"})


    # Graph 1: LinkedIn usage by age grup
    st.subheader("LinkedIn Usage by Age Group")

     # Create age groups
    li_df["age_group"] = pd.cut(
        li_df["age"],
        bins=[18, 24, 34, 44, 54, 64, 120],
        labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
    )

    # Calculate usage rate
    age_rates = (
        li_df.groupby("age_group")["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    # Custom color palette for aesthetic purposes
    linkedin_lilac_palette = [
        "#0A66C2",  
        "#7FB8F6",  
        "#C7A0E8",  
        "#E6D7FF",  
        "#5886B0",  
        "#DCC7FF"   
    ]

    age_chart = alt.Chart(age_rates).mark_bar().encode(
        x=alt.X("age_group:N", title="Age group"),
        y=alt.Y("linkedin:Q", title="LinkedIn usage (%)"),
        color=alt.Color(
            "age_group:N",
            title="Age group",
            scale=alt.Scale(range=linkedin_lilac_palette)
        ),
        tooltip=[
            alt.Tooltip("age_group:N", title="Age group"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn users", format=".1f")
        ]
    ).properties(
        width=600,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(age_chart, use_container_width=True)

    # Graph 2: Usage by income level
    st.subheader("LinkedIn Usage by Income Level")

    # Income Labels
    income_labels = {
        1: "< $10K",
        2: "$10K–$20K",
        3: "$20K–$30K",
        4: "$30K–$40K",
        5: "$40K–$50K",
        6: "$50K–$75K",
        7: "$75K–$100K",
        8: "$100K–$150K",
        9: "$150K+"
    }

    li_df["income_label"] = li_df["income"].map(income_labels)

    # LI usage by income
    income_usage = (
        li_df.groupby("income_label")["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    # income order verification
    income_order = list(income_labels.values())

    income_palette = [
        "#C7A0E8",  # lilac light
        "#A985DD",
        "#8F6FD3",
        "#7A5BC8",
        "#5E4CBF",
        "#4F6DBE",
        "#3E86C4",
        "#1D66C2",  # dark LinkedIn blue
        "#0A4EA0"   # deeper blue for high incomes
    ]

    # Bar chart with multiple colors
    income_bar = alt.Chart(income_usage).mark_bar(
        opacity=0.85
    ).encode(
        x=alt.X("income_label:N", sort=income_order, title="Income Level"),
        y=alt.Y("linkedin:Q", title="LinkedIn Usage (%)"),
        color=alt.Color(
            "income_label:N",
            sort=income_order,
            scale=alt.Scale(range=income_palette),
            title="Income Level"
        ),
        tooltip=[
            alt.Tooltip("income_label:N", title="Income"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn users", format=".1f")
        ]
    ).properties(
        width=650,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(income_bar, use_container_width=True)


    #Graph 3: Usage based on parental status
    # Graph: LinkedIn Usage by Parental Status
    st.subheader("LinkedIn Usage by Parental Status")

    # Convert parent variable into readable labels
    li_df["parent_label"] = li_df["parent"].map({0: "Non-Parent", 1: "Parent"})

    # Calculate LI usage by parental status
    parent_usage = (
        li_df.groupby("parent_label")["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    # Color palette (matches your theme)
    parent_palette = ["#C7A0E8", "#0A66C2"]  # lilac + LinkedIn blue

    # Bar chart
    parent_bar = alt.Chart(parent_usage).mark_bar().encode(
        x=alt.X("parent_label:N", title="Parental Status"),
        y=alt.Y("linkedin:Q", title="LinkedIn Usage (%)"),
        color=alt.Color(
            "parent_label:N",
            scale=alt.Scale(range=parent_palette),
            title="Parental Status"
        ),
        tooltip=[
            alt.Tooltip("parent_label:N", title="Parental Status"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn Users", format=".1f")
        ]
    ).properties(
        width=450,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(strokeWidth=0)

    st.altair_chart(parent_bar, use_container_width=True)


    # Graph 4: LI usage by age group and gender 
    st.subheader("LinkedIn Usage by Age Group and Gender")

    li_df["gender_label"] = li_df["female"].map({0: "Male", 1: "Female"})

    # LI usage by age group and gender
    age_gender_rates = (
        li_df.groupby(["age_group", "gender_label"])["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    gender_palette = ["#0A66C2", "#C7A0E8"]

    age_gender_chart = alt.Chart(age_gender_rates).mark_line(point=True).encode(
        x=alt.X("age_group:N", title="Age group"),
        y=alt.Y("linkedin:Q", title="LinkedIn usage (%)"),
        color=alt.Color(
            "gender_label:N",
            title="Gender",
            scale=alt.Scale(range=gender_palette)
        ),
        tooltip=[
            alt.Tooltip("age_group:N", title="Age group"),
            alt.Tooltip("gender_label:N", title="Gender"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn users", format=".1f")
        ]
    ).properties(
        width=650,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(age_gender_chart, use_container_width=True)

    # Graph 5: LI usage by education level and gender
    st.subheader("LinkedIn Usage by Education Level and Gender")

    # Label education levels
    education_labels = {
        1: "Less than HS",
        2: "HS incomplete",
        3: "HS graduate",
        4: "Some college",
        5: "College graduate",
        6: "Post-college",
        7: "Professional degree",
        8: "Doctorate"
    }
    li_df["education_label"] = li_df["education"].map(education_labels)

    # Gender labels
    li_df["gender_label"] = li_df["female"].map({0: "Male", 1: "Female"})

    # LI usage grouped by education and gender
    edu_gender_rates = (
        li_df.groupby(["education_label", "gender_label"])["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    # Colors for gender
    gender_palette = ["#0A66C2", "#C7A0E8"]  # adjust which is male/female if you want

    edu_gender_chart = alt.Chart(edu_gender_rates).mark_line(point=True).encode(
        x=alt.X("education_label:N", title="Education level"),
        y=alt.Y("linkedin:Q", title="LinkedIn usage (%)"),
        color=alt.Color(
            "gender_label:N",
            title="Gender",
            scale=alt.Scale(range=gender_palette)
        ),
        tooltip=[
            alt.Tooltip("education_label:N", title="Education"),
            alt.Tooltip("gender_label:N", title="Gender"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn users", format=".1f")
        ]
    ).properties(
        width=650,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(edu_gender_chart, use_container_width=True)

    #Graph 6: Heatmap from EDA in jupyter notebook

    st.subheader("LinkedIn Usage by Income and Education Level")
    
    # LI usagefor each income × education bin
    heat_data = (
        li_df.groupby(["education_label", "income_label"])["linkedin"]
        .mean()
        .mul(100)
        .reset_index()
    )

    # Heatmap colors
    heat_palette = ["#E6D7FF", "#C7A0E8", "#7FB8F6", "#0A66C2"]

    heatmap = alt.Chart(heat_data).mark_rect().encode(
        x=alt.X("income_label:N", title="Income level"),
        y=alt.Y("education_label:N", title="Education level"),
        color=alt.Color(
            "linkedin:Q",
            title="LinkedIn usage (%)",
            scale=alt.Scale(range=heat_palette)
        ),
        tooltip=[
            alt.Tooltip("education_label:N", title="Education"),
            alt.Tooltip("income_label:N", title="Income"),
            alt.Tooltip("linkedin:Q", title="% LinkedIn users", format=".1f")
        ]
    ).properties(
        width=650,
        height=350
    ).configure_axis(
        labelColor="#1D2226",
        titleColor="#0A66C2"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(heatmap, use_container_width=True)








