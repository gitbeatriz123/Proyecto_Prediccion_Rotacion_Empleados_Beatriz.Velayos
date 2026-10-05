# Procedencia y uso de los datos

## Dataset principal

El proyecto utiliza una copia local del conjunto conocido como **IBM HR Analytics Employee Attrition & Performance**, con 1.470 registros y 35 variables.

La identificación del conjunto se hace por coincidencia de nombre, estructura y contenido del ejercicio. El repositorio no afirma que los registros representen una plantilla real de IBM ni de ninguna otra empresa.

**Uso en este proyecto:** educativo y demostrativo.

Antes de redistribuir la copia de datos por separado, debe comprobarse la licencia y las condiciones aplicables a la fuente concreta utilizada para obtenerla.

## Encuesta de clima

'encuesta_clima.csv' es **100 % sintética**. Se genera con 'scripts/generate_synthetic_survey.py', semilla 42 y cinco escalas aleatorias de 1 a 5. El generador no consulta 'Attrition'.

Por tanto, las variables 'survey_*' no representan respuestas reales, mediciones observadas ni una encuesta validada. Su función es demostrar integración de fuentes y flujo ETL.

## Uso responsable

El proyecto no debe utilizarse para evaluar, seleccionar, despedir, promocionar o retener personas. Las asociaciones del modelo no son relaciones causales y los resultados no están validados para una organización real.
