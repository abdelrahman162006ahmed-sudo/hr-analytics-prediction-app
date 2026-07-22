import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE
from xgboost.callback import EarlyStopping
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,VotingClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay



# data=pd.read_csv('Employee.csv')

# data = data.drop(columns=["City"])


# employee_columns = ["Education", "Gender", "EverBenched"]
# employee_encoders = {}

# for col in employee_columns:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     employee_encoders[col] = le


# data["JoiningYear"] = data["JoiningYear"] + 8

# # Features & Target
# X = data.drop("LeaveOrNot", axis=1)
# y = data["LeaveOrNot"]


# employee_feature_names = X.columns.tolist()

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y,
# )

# # SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# # Scaling
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # XGBoost
# xgb_model = xgb.XGBClassifier(
#     max_depth=5,
#     learning_rate=0.01,
#     n_estimators=300,
#     subsample=0.5,
#     colsample_bytree=0.8,
#     scale_pos_weight=1,
#     random_state=42,
#     eval_metric="logloss",
# )

# # Random Forest
# rf_model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=5,
#     min_samples_split=10,
#     min_samples_leaf=10,
#     random_state=42,
# )

# # Train
# xgb_model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)

# # Voting
# leave_model = VotingClassifier(
#     estimators=[
#         ("xgb", xgb_model),
#         ("rf", rf_model),
#     ],
#     voting="soft",
# )

# leave_model.fit(X_train, y_train)

# # Prediction
# leave_pred = leave_model.predict(X_test) 
# joblib.dump(leave_model,"leave_model.pkl")
# joblib.dump(scaler,"scaler_leave.pkl")
# joblib.dump(employee_encoders,"employee_encoders.pkl")
# #####################################################################

# data2=pd.read_csv('salary.csv')




# data2["location"] = data2["location"].replace("India", "Egypt")

# salary_columns = [
#     "job_title",
#     "education_level",
#     "industry",
#     "company_size",
#     "remote_work",
#     "location",
# ]

# salary_encoders = {}

# for col in salary_columns:

#     le = LabelEncoder()
#     data2[col] = le.fit_transform(data2[col])
#     salary_encoders[col] = le

# X2 = data2.drop("salary", axis=1)
# y2 = data2["salary"]

# salary_feature_names = X2.columns.tolist()

# X_train2, X_test2, y_train2, y_test2 = train_test_split(
#     X2,
#     y2,
#     test_size=0.20,
#     random_state=42,
# )

# salary_model = xgb.XGBRegressor(
#     max_depth=7,
#     learning_rate=0.1,
#     n_estimators=500,
#     subsample=0.7,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1,
# )

# salary_model.fit(X_train2, y_train2)

# salary_pred = salary_model.predict(X_test2)
# joblib.dump(salary_model,"salary_model.pkl")
# joblib.dump(salary_encoders,"salary_encoders.pkl")

# ===========================
# Load Models
# ===========================

leave_model = joblib.load("leave_model.pkl")
salary_model = joblib.load("salary_model.pkl")

scaler = joblib.load("scaler_leave.pkl")

employee_encoders = joblib.load("encoder_leave.pkl")
salary_encoders = joblib.load("encoder_salary.pkl")

# ===========================
# Page Config
# ===========================k


st.set_page_config(
    page_title="Employee AI System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Employee AI System")

tab1, tab2 = st.tabs([
    "Leave Prediction",
    "Salary Prediction"
])


# ====================================================
#               LEAVE PREDICTION
# ====================================================

with tab1:

    st.header("🏢 Employee Leave Prediction")
    st.write("Enter the employee information below.")

    col1, col2 = st.columns(2)

    with col1:

        leave_education = st.selectbox(
            "Education",
            employee_encoders["Education"].classes_
        )

        joining_year = st.number_input(
            "Joining Year",
            min_value=2020,
            max_value=2026,
            
        )

        payment_tier = st.selectbox(
            "Payment Tier",
            [1, 2, 3]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            employee_encoders["Gender"].classes_
        )

        experience = st.number_input(
            "Experience In Current Domain(Tier)",
            min_value=0,
            max_value=7,
            
        )

        everbenched = st.selectbox(
            "Ever Benched",
            employee_encoders["EverBenched"].classes_
        )

    if st.button(
        "🔍 Predict Leave",
        use_container_width=True
    ):

        # ===========================
        # Validation
        # ===========================

        current_year = 2026
        years_since_joining = current_year - joining_year

        # if experience > years_since_joining:
        #     st.warning(
        #         "Experience cannot be greater than the years since joining the company."
        #     )
        #     st.stop()

        if age < 22 and experience > 5:
            st.warning(
                "The entered age and experience seem inconsistent."
            )
            st.stop()

        if age < experience + 18:
            st.warning(
                "Age is not consistent with the entered experience."
            )
            st.stop()

        # ===========================
        # Encoding
        # ===========================

        education_encoded = employee_encoders["Education"].transform([leave_education])[0]
        gender_encoded = employee_encoders["Gender"].transform([gender])[0]
        benched_encoded = employee_encoders["EverBenched"].transform([everbenched])[0]

        joining_year_model = joining_year 

        input_data = pd.DataFrame({
            "Education": [education_encoded],
            "JoiningYear": [joining_year_model],
            "PaymentTier": [payment_tier],
            "Age": [age],
            "Gender": [gender_encoded],
            "EverBenched": [benched_encoded],
            "ExperienceInCurrentDomain": [experience]
        })

        # ===========================
        # Scaling
        # ===========================

        input_scaled = scaler.transform(input_data)

        # ===========================
        # Prediction
        # ===========================

        with st.spinner("Predicting..."):

            prediction = leave_model.predict(input_scaled)[0]
            probability = leave_model.predict_proba(input_scaled)[0]

        st.divider()

        if prediction == 1:

            st.error("⚠️ Employee is likely to leave.")

            st.metric(
                "Confidence",
                f"{probability[1]*100:.2f}%"
            )

        else:

            st.success("✅ Employee is likely to stay.")

            st.metric(
                "Confidence",
                f"{probability[0]*100:.2f}%"
            )

            st.balloons()

# ====================================================
#               SALARY PREDICTION
# ====================================================

with tab2:

    st.header("💰 Salary Prediction")
    st.write("Enter the employee information below.")

    col1, col2 = st.columns(2)

    with col1:

        experience_years = st.number_input(
            "Experience Years",
            min_value=0,
            max_value=25,
            
        )

        skills_count = st.number_input(
            "Skills Count",
            min_value=0,
            max_value=19,
            
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=5,
            
        )

        education = st.selectbox(
            "Education Level",
            salary_encoders["education_level"].classes_
        )

        job_title = st.selectbox(
            "Job Title",
            salary_encoders["job_title"].classes_
        )






    with col2:

        industry = st.selectbox(
            "Industry",
            salary_encoders["industry"].classes_
        )

        company = st.selectbox(
            "Company Size",
            salary_encoders["company_size"].classes_
        )

        location = st.selectbox(
            "Location",
            salary_encoders["location"].classes_
        )

        remote = st.selectbox(
            "Remote Work",
            salary_encoders["remote_work"].classes_
        )

    if st.button(
        "💲 Predict Salary",
        use_container_width=True
    ):

        # ===========================
        # Validation
        # ===========================

        if experience_years > 10 and skills_count == 0:
            st.warning(
                "A person with more than 10 years of experience is unlikely to have 0 skills."
            )
            st.stop()

        if certifications > skills_count:
            st.warning(
                "Certifications cannot be greater than the number of skills."
            )
            st.stop()

        if experience_years == 0 and certifications == 5:
            st.warning(
                "Too many certifications for someone with no experience."
            )
            st.stop()
            

        # ===========================
        # Encoding
        # ===========================

        job_title_encoded = salary_encoders["job_title"].transform([job_title])[0]
        education_encoded = salary_encoders["education_level"].transform([education])[0]
        industry_encoded = salary_encoders["industry"].transform([industry])[0]
        company_encoded = salary_encoders["company_size"].transform([company])[0]
        location_encoded = salary_encoders["location"].transform([location])[0]
        remote_encoded = salary_encoders["remote_work"].transform([remote])[0]

        input_data = pd.DataFrame({
            "job_title": [job_title_encoded],
            "experience_years": [experience_years],
            "education_level": [education_encoded],
            "skills_count": [skills_count],
            "industry": [industry_encoded],
            "company_size": [company_encoded],
            "location": [location_encoded],
            "remote_work": [remote_encoded],
            "certifications": [certifications]
        })

       

        with st.spinner("Predicting Salary..."):

            salary = salary_model.predict(input_data)[0]

            st.divider()

            st.success("✅ Salary Prediction Completed")

            st.metric(
                label="Estimated Salary",
           value=f"${salary:,.2f}"
      )


        st.balloons()

        # ===========================
        # Salary Category
        # ===========================

        if salary < 50000:
            st.info("💼 Salary Level: Entry Level")

        elif salary < 100000:
            st.info("💼 Salary Level: Mid Level")

        elif salary < 150000:
            st.info("💼 Salary Level: Senior Level")

        else:
            st.success("🏆 Salary Level: Executive")


        st.subheader("Employee Summary")

        st.write(f"**Job Title:** {job_title}")
        st.write(f"**Education:** {education}")
        st.write(f"**Experience:** {experience_years} Years")
        st.write(f"**Skills:** {skills_count}")
        st.write(f"**Industry:** {industry}")
        st.write(f"**Company Size:** {company}")
        st.write(f"**Location:** {location}")


