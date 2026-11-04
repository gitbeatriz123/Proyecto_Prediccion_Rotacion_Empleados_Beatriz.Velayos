# Procedencia y uso de los datos

## Dataset principal

El proyecto utiliza una copia local del conjunto conocido como **IBM HR Analytics Employee Attrition & Performance**, con 1.470 registros y 35 variables. Incluye datos de departamento, puesto, antigüedad, viajes, horas extra y otras características laborales.

Antes de redistribuir la copia de datos por separado, deben comprobarse la licencia y las condiciones aplicables a la fuente concreta utilizada para obtenerla.

## Encuesta de clima

`encuesta_clima.csv` es **100 % sintética**. Se genera con `scripts/generate_synthetic_survey.py`, semilla 42 y cinco escalas aleatorias de 1 a 5. El generador no consulta la etiqueta `Attrition`.

Las variables `survey_*` se excluyen del modelado. La encuesta permite recorrer la integración de dos fuentes y el flujo ETL; sus puntuaciones no representan respuestas de empleados reales.

## Lectura de resultados

El panel HTML y los informes separan las tasas observadas de las métricas de evaluación y las señales OOF. Cada indicador debe interpretarse con su definición y denominador.
