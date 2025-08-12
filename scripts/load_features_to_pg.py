import argparse, pandas as pd, numpy as np, psycopg2
from psycopg2.extras import execute_values
ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True); ap.add_argument("--pg_host",default="postgres")
ap.add_argument("--pg_db",default="mlops"); ap.add_argument("--pg_user",default="ml"); ap.add_argument("--pg_pass",default="ml")
a=ap.parse_args()
df=pd.read_parquet(a.input)
cols=["EmployeeNumber","Department","JobRole","Gender","Age","MonthlyIncome","YearsAtCompany","OverTime",
      "survey_Engagement","survey_Satisfaction","survey_WorkLifeBalanceSurvey","survey_ManagerRelationship",
      "survey_RemoteWorkSatisfaction","overtime_flag","income_yearly","tenure_ratio"]
cols=[c for c in cols if c in df.columns]
dim=df[cols].copy(); dim["EmployeeNumber"]=dim["EmployeeNumber"].astype(int)
for c in ["Age","MonthlyIncome","YearsAtCompany","survey_Engagement","survey_Satisfaction",
          "survey_WorkLifeBalanceSurvey","survey_ManagerRelationship","survey_RemoteWorkSatisfaction","overtime_flag"]:
    if c in dim.columns: dim[c]=pd.to_numeric(dim[c], errors="coerce")
for c in ["income_yearly","tenure_ratio"]:
    if c in dim.columns: dim[c]=pd.to_numeric(dim[c], errors="coerce")
conn=psycopg2.connect(host=a.pg_host,dbname=a.pg_db,user=a.pg_user,password=a.pg_pass); cur=conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS employees_features(
  employee_number INT PRIMARY KEY,
  department TEXT, jobrole TEXT, gender TEXT, age INT,
  monthlyincome DOUBLE PRECISION, yearsatcompany INT, overtime TEXT,
  survey_engagement INT, survey_satisfaction INT, survey_worklifebalancesurvey INT,
  survey_managerrelationship INT, survey_remoteworksatisfaction INT,
  overtime_flag INT, income_yearly DOUBLE PRECISION, tenure_ratio DOUBLE PRECISION
);"""); conn.commit()
cur.execute("TRUNCATE employees_features;"); conn.commit()
cols_db=["employee_number","department","jobrole","gender","age","monthlyincome","yearsatcompany","overtime",
         "survey_engagement","survey_satisfaction","survey_worklifebalancesurvey","survey_managerrelationship",
         "survey_remoteworksatisfaction","overtime_flag","income_yearly","tenure_ratio"]
def row(r):
    g=lambda k: r.get(k)
    return (int(r["EmployeeNumber"]), g("Department"), g("JobRole"), g("Gender"),
            None if "Age" not in r or pd.isna(r["Age"]) else int(r["Age"]),
            g("MonthlyIncome"),
            None if "YearsAtCompany" not in r or pd.isna(r["YearsAtCompany"]) else int(r["YearsAtCompany"]),
            g("OverTime"), g("survey_Engagement"), g("survey_Satisfaction"),
            g("survey_WorkLifeBalanceSurvey"), g("survey_ManagerRelationship"),
            g("survey_RemoteWorkSatisfaction"), g("overtime_flag"), g("income_yearly"), g("tenure_ratio"))
rows=[row(rec) for rec in dim.to_dict(orient="records")]
execute_values(cur, f"INSERT INTO employees_features({','.join(cols_db)}) VALUES %s", rows, page_size=500)
conn.commit(); cur.close(); conn.close()
print(f"Cargadas {len(rows)} filas en employees_features.")
