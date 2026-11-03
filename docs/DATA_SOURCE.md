# Procedencia y uso de los datos

## Dataset principal

El proyecto utiliza una copia local del conjunto conocido como **IBM HR Analytics Employee Attrition & Performance**, con 1.470 registros y 35 variables.


Antes de redistribuir la copia de datos por separado, debe comprobarse la licencia y las condiciones aplicables a la fuente concreta utilizada para obtenerla.

## Encuesta de clima

'encuesta_clima.csv' es **100 % sintética**. Se genera con 'scripts/generate_synthetic_survey.py', semilla 42 y cinco escalas aleatorias de 1 a 5. El generador no consulta 'Attrition'.

Las variables 'survey_*' se excluyen del modelado; el generador permite recorrer la integración de datos y el flujo ETL.

## Lectura de los datos

Las tasas observadas y las puntuaciones de los modelos se presentan por separado en los informes y en Power BI para orientar diagnósticos y seguimiento por área.
