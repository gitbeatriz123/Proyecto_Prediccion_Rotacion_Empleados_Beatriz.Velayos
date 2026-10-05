CREATE TABLE IF NOT EXISTS metrics (
    model_name TEXT PRIMARY KEY,
    roc_auc DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,
    f1_opt DOUBLE PRECISION,
    thr_opt DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    employee_number INTEGER NOT NULL,
    proba DOUBLE PRECISION NOT NULL,
    pred INTEGER NOT NULL CHECK (pred IN (0, 1)),
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (employee_number, model_name),
    CONSTRAINT predictions_model_fk FOREIGN KEY (model_name) REFERENCES metrics(model_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_employee ON predictions(employee_number);

CREATE TABLE IF NOT EXISTS employees_features (
    employee_number INTEGER PRIMARY KEY,
    department TEXT, jobrole TEXT, gender TEXT, age INTEGER,
    monthlyincome DOUBLE PRECISION, yearsatcompany INTEGER, overtime TEXT,
    survey_engagement INTEGER, survey_satisfaction INTEGER,
    survey_worklifebalancesurvey INTEGER, survey_managerrelationship INTEGER,
    survey_remoteworksatisfaction INTEGER, overtime_flag INTEGER,
    income_yearly DOUBLE PRECISION, tenure_ratio DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS feature_effects (
    feature TEXT NOT NULL, value DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW predictions_latest AS
SELECT p.employee_number, p.proba, p.pred, p.model_name, p.created_at FROM predictions p;

CREATE OR REPLACE VIEW v_kpi_dept AS
SELECT e.employee_number, e.department, p.model_name, 1::INTEGER AS n,
       AVG(p.pred) OVER (PARTITION BY p.model_name, e.department) AS risk_rate,
       AVG(p.proba) OVER (PARTITION BY p.model_name, e.department) AS avg_proba
FROM employees_features e JOIN predictions_latest p USING (employee_number);

CREATE OR REPLACE VIEW v_kpi_jobrole AS
SELECT e.employee_number, e.jobrole, p.model_name, 1::INTEGER AS n,
       AVG(p.pred) OVER (PARTITION BY p.model_name, e.jobrole) AS risk_rate,
       AVG(p.proba) OVER (PARTITION BY p.model_name, e.jobrole) AS avg_proba
FROM employees_features e JOIN predictions_latest p USING (employee_number);

CREATE OR REPLACE VIEW v_kpi_overtime AS
SELECT e.employee_number, e.overtime, p.model_name, 1::INTEGER AS n,
       AVG(p.pred) OVER (PARTITION BY p.model_name, e.overtime) AS risk_rate,
       AVG(p.proba) OVER (PARTITION BY p.model_name, e.overtime) AS avg_proba
FROM employees_features e JOIN predictions_latest p USING (employee_number);

CREATE OR REPLACE VIEW v_kpi_tenure_band AS
SELECT e.employee_number,
       CASE WHEN e.yearsatcompany < 2 THEN '0–2'
            WHEN e.yearsatcompany < 5 THEN '2–5'
            WHEN e.yearsatcompany < 10 THEN '5–10'
            ELSE '>=10' END AS tenure_band,
       p.model_name, 1::INTEGER AS n,
       AVG(p.pred) OVER (PARTITION BY p.model_name,
         CASE WHEN e.yearsatcompany < 2 THEN '0–2'
              WHEN e.yearsatcompany < 5 THEN '2–5'
              WHEN e.yearsatcompany < 10 THEN '5–10'
              ELSE '>=10' END) AS risk_rate,
       AVG(p.proba) OVER (PARTITION BY p.model_name,
         CASE WHEN e.yearsatcompany < 2 THEN '0–2'
              WHEN e.yearsatcompany < 5 THEN '2–5'
              WHEN e.yearsatcompany < 10 THEN '5–10'
              ELSE '>=10' END) AS avg_proba
FROM employees_features e JOIN predictions_latest p USING (employee_number);
