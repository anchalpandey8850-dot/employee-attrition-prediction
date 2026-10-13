
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_pipeline.pkl")


pipeline = load_model()

st.title("Employee Attrition Prediction")
st.write(
    "Enter employee details to predict the likelihood of attrition."
)

st.warning(
    "This prediction is an estimate, not a guarantee. "
    "Do not use it as the sole basis for employment decisions."
)

with st.form("employee_form"):

    st.subheader("Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 60, 30)

        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )

        daily_rate = st.number_input("Daily Rate", 102, 1499, 800)

        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )

        distance = st.number_input("Distance From Home", 1, 29, 5)
        education = st.selectbox("Education", [1, 2, 3, 4, 5])

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences", "Other", "Medical", "Marketing",
                "Technical Degree", "Human Resources"
            ]
        )

        environment = st.selectbox(
            "Environment Satisfaction", [1, 2, 3, 4]
        )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
        involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive", "Research Scientist",
                "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director",
                "Human Resources"
            ]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction", [1, 2, 3, 4]
        )

        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )

        monthly_income = st.number_input(
            "Monthly Income", 1009, 19999, 5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate", 2094, 26999, 14000
        )

    with col3:
        companies = st.number_input("Companies Worked", 0, 9, 2)
        overtime = st.selectbox("Overtime", ["Yes", "No"])

        salary_hike = st.number_input(
            "Percent Salary Hike", 11, 25, 15
        )

        performance = st.selectbox("Performance Rating", [3, 4])

        relationship = st.selectbox(
            "Relationship Satisfaction", [1, 2, 3, 4]
        )

        stock_options = st.selectbox(
            "Stock Option Level", [0, 1, 2, 3]
        )

        total_working_years = st.number_input(
            "Total Working Years", 0, 40, 8
        )

        training = st.number_input(
            "Training Times Last Year", 0, 6, 3
        )

        work_life = st.selectbox(
            "Work-Life Balance", [1, 2, 3, 4]
        )

        years_company = st.number_input("Years at Company", 0, 40, 5)
        years_role = st.number_input("Years in Current Role", 0, 18, 3)

        years_promotion = st.number_input(
            "Years Since Last Promotion", 0, 15, 1
        )

        years_manager = st.number_input(
            "Years With Current Manager", 0, 17, 3
        )

    submitted = st.form_submit_button("Predict Attrition")


if submitted:

    employee = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": companies,
        "OverTime": overtime,
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "StockOptionLevel": stock_options,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager
    }])

    try:
        prediction = pipeline.predict(employee)[0]
        probabilities = pipeline.predict_proba(employee)[0]

        # Find the probability corresponding to Attrition = Yes (class 1)
        classes = list(pipeline.classes_)
        yes_index = classes.index(1)
        probability = probabilities[yes_index]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Predicted Attrition: Yes")
        else:
            st.success("Predicted Attrition: No")

        st.metric(
            "Estimated Attrition Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

    except Exception as error:
        st.error(f"Prediction failed: {error}")
