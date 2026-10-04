#!/usr/bin/env python3
"""Generate a reproducible synthetic climate survey from a fixed seed."""
import argparse
import csv
import random
from pathlib import Path

SCORE_COLUMNS = (
    "Engagement",
    "Satisfaction",
    "WorkLifeBalanceSurvey",
    "ManagerRelationship",
    "RemoteWorkSatisfaction",
)


def generate(input_path: Path, output_path: Path, seed: int = 42) -> int:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "EmployeeNumber" not in reader.fieldnames:
            raise ValueError("El CSV HR debe incluir EmployeeNumber.")
        ids = [row["EmployeeNumber"].strip() for row in reader]
    if not ids or any(not value for value in ids):
        raise ValueError("EmployeeNumber no puede estar vacío.")
    if len(ids) != len(set(ids)):
        raise ValueError("EmployeeNumber debe ser único en el CSV HR.")

    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("EmployeeNumber", *SCORE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for employee_id in ids:
            writer.writerow({
                "EmployeeNumber": employee_id,
                **{column: rng.randint(1, 5) for column in SCORE_COLUMNS},
            })
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV HR con EmployeeNumber")
    parser.add_argument("--output", required=True, help="Ruta del CSV sintético de encuesta")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = generate(Path(args.input), Path(args.output), args.seed)
    print(f"Generadas {count} filas sintéticas; escala independiente 1–5; seed={args.seed}.")


if __name__ == "__main__":
    main()
