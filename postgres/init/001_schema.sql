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
