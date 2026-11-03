# Actualizar el panel de Power BI

El archivo PBIX contiene el diseño y las medidas del panel. Para que muestre la ejecución canónica actual hay que regenerar y persistir los resultados y actualizar sus conexiones en Power BI Desktop.

## 1. Regenerar y cargar los resultados actuales

Desde la raíz del repositorio, sigue el bloque **Pipeline canónico** del README. Incluye el control de datos, la integración con PySpark, el entrenamiento de los tres modelos, la carga de características y la persistencia de LogReg, RF y MLP en PostgreSQL.

Comprueba que Docker Compose está en marcha y que las variables de PostgreSQL coinciden con las de `.env`. El puerto publicado por defecto es `5442`; la base de datos predeterminada es `mlops`. Si no has cambiado las variables, el usuario predeterminado es `ml`; revisa la contraseña configurada localmente. No publiques el archivo `.env`.

Regenera también las figuras y la copia del CSV de coeficientes:

```bash
docker compose run --rm jupyter python /scripts/render_dashboard_previews.py
```

## 2. Corregir las conexiones de PostgreSQL

Abre `bi/Panel_Rotacion_Empleados.pbix` con Power BI Desktop. En **Archivo → Opciones y configuración → Configuración de origen de datos**, localiza el origen PostgreSQL actual, que apunta a `192.168.0.13:5442`, y cambia el servidor al equipo donde se ejecuta Docker. Si Power BI Desktop y Docker están en el mismo equipo, usa `localhost:5442`; si están en equipos distintos, utiliza la dirección local accesible del equipo Docker.

Mantén la base de datos `mlops`. En la autenticación de PostgreSQL, introduce las credenciales que estén configuradas en el `.env` local. Aplica el cambio al origen existente para que las consultas de las tablas `public` usen el mismo servidor.

## 3. Actualizar el CSV de factores asociados

En **Transformar datos**, selecciona la consulta `feature_effects` y abre **Editor avanzado**. Sustituye la ruta absoluta local antigua por la ruta completa a `docs/powerbi/feature_effects.csv` dentro de tu copia del proyecto. Esa copia se genera con el comando del paso 1.

La ruta es local al equipo donde se abre Power BI. Conserva las dos columnas `feature` y `effect`. Como el CSV está codificado en UTF-8, usa `Encoding=65001` en `Csv.Document` para que se vean correctamente las tildes. La consulta puede quedar así, sustituyendo la ruta de ejemplo por la ruta completa de tu copia local:

```powerquery
let
    Origen = Csv.Document(
        File.Contents("C:\\ruta\\al\\proyecto\\docs\\powerbi\\feature_effects.csv"),
        [Delimiter=",", Columns=2, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Encabezados = Table.PromoteHeaders(Origen, [PromoteAllScalars=true]),
    Tipos = Table.TransformColumnTypes(Encabezados, {{"feature", type text}, {"effect", type number}})
in
    Tipos
```

Al guardar el PBIX, la ruta nueva queda asociada al archivo.

## 4. Incluir MLP en el selector del panel

El selector desconectado `Dim Modelo` del PBIX contiene solo `LogReg` y `RF`. En **Transformar datos**, selecciona `Dim Modelo`, abre el **Editor avanzado** y reemplaza su contenido por:

```powerquery
let
    Origen = #table(
        type table [model_name = text],
        {{"LogReg"}, {"RF"}, {"MLP"}}
    )
in
    Origen
```

Así el selector permite consultar las métricas y predicciones de los tres modelos que se persisten en PostgreSQL.

## 5. Alinear los rótulos del panel con lo que mide

Las tablas de predicciones contienen `proba` (probabilidad estimada) y `pred` (clasificación binaria al umbral del modelo). Las medidas del PBIX que promedian `pred` muestran la proporción de empleados que el modelo clasifica como señal; no calculan la tasa observada de salidas.

En **Vista de modelo**, renombra las medidas `Churn Rate Pred (Sel Model)`, `Churn Rate Dept (Sel Model)` y `Churn Rate JobRole (Sel Model)` como **Riesgo predicho (%)**, **Riesgo predicho por departamento (%)** y **Riesgo predicho por puesto (%)**. Actualiza sus títulos y el texto del tooltip: cambia “tasa de rotación” o “Churn estimado” por **riesgo predicho**. Mantén las tasas observadas etiquetadas como observadas y con su denominador.

## 6. Actualizar y comprobar

Pulsa **Cerrar y aplicar** y después **Actualizar**. Comprueba estos puntos antes de guardar:

- `public metrics` muestra tres modelos, con LogReg: ROC-AUC 0,823 / PR-AUC 0,575 / F1 evaluado 0,542; RF: 0,803 / 0,540 / 0,523; y MLP: 0,814 / 0,559 / 0,522.
- El selector ofrece LogReg, RF y MLP.
- Las páginas muestran los valores del modelo seleccionado y no quedan errores de conexión o de ruta del CSV.
- Los rótulos distinguen el **riesgo predicho por el modelo final**, las **tasas observadas** y las **señales OOF** de la evaluación. Son salidas diferentes.
- El panel y la presentación conservan el nombre **Random Forest**.

Guarda el archivo actualizado en `bi/Panel_Rotacion_Empleados.pbix`. Si trabajas en otra copia, reemplaza el PBIX del repositorio con la versión ya actualizada.
