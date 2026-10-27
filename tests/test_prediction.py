from utils.prediction import predict_stage

patient = {

    "Country":"India",

    "Age":50,

    "Gender":"Female",

    "Smoking_History":0,

    "Obesity":0,

    "Diabetes":1,

    "Chronic_Pancreatitis":0,

    "Family_History":1,

    "Hereditary_Condition":0,

    "Jaundice":1,

    "Abdominal_Discomfort":1,

    "Back_Pain":0,

    "Weight_Loss":1,

    "Development_of_Type2_Diabetes":0,

    "Survival_Time_Months":12,

    "Treatment_Type":"Chemotherapy",

    "Survival_Status":0,

    "Alcohol_Consumption":0,

    "Physical_Activity_Level":"Medium",

    "Diet_Processed_Food":"High",

    "Access_to_Healthcare":"Medium",

    "Urban_vs_Rural":"Urban",

    "Economic_Status":"Middle"

}

stage, confidence = predict_stage(patient)

print(stage)

print(confidence)