# Crear el PBIX actualizado en Power BI Desktop

El libro [Panel_PowerBI_datos_canonicos.xlsx](../../bi/Panel_PowerBI_datos_canonicos.xlsx) contiene los resultados canónicos en tablas que Power BI Desktop puede importar directamente. No requiere Docker, PostgreSQL ni programación.

## 1. Descargar y guardar los archivos

1. Descarga el libro Excel desde la carpeta `bi/` del repositorio.
2. Guarda el libro en una carpeta de tu ordenador Windows.
3. Abre Power BI Desktop.

## 2. Importar los datos

1. En Power BI Desktop, pulsa **Inicio → Obtener datos → Libro de Excel**.
2. Selecciona `Panel_PowerBI_datos_canonicos.xlsx` y pulsa **Abrir**.
3. En la ventana **Navegador**, marca estas hojas:
   - `Metricas_modelos`
   - `Departamentos`
   - `Horas_extra`
   - `Factores_LogReg`
   - `Recomendaciones`
4. Puedes dejar `Notas` sin marcar; contiene definiciones y fuentes.
5. Pulsa **Cargar**.

## 3. Crear las páginas del informe

### Resumen de modelos

- Añade una tabla con `Modelo`, `ROC_AUC`, `PR_AUC`, `F1_evaluado` y `Umbral_referencia`.
- Añade un gráfico de columnas agrupadas. Usa `Modelo` como eje y las métricas `ROC_AUC`, `PR_AUC` y `F1_evaluado` como valores.
- Mantén el título **Comparación de modelos**. El nombre del modelo es **Random Forest**.

### Departamentos

- Añade un gráfico de barras con `Departamento` y `Tasa_observada`.
- Añade una tabla con `Departamento`, `Empleados`, `Salidas_observadas` y `Senales_OOF_LogReg`.
- Titula las señales como **Señales OOF de Regresión Logística (umbral 0,706)**.

### Horas extra

- Añade un gráfico de columnas con `Grupo` y `Tasa_observada`.
- Añade `Empleados` y `Salidas_observadas` en una tabla junto al gráfico.

### Factores y recomendaciones

- Añade un gráfico de barras con `Factor` y `Coeficiente`. Usa los diez primeros factores de la tabla, que ya están ordenados por magnitud.
- Añade una tabla con los campos de `Recomendaciones`.
- Titula el gráfico **Factores asociados en Regresión Logística**. Los coeficientes describen asociaciones del modelo.

## 4. Guardar el PBIX

1. Pulsa **Archivo → Guardar como**.
2. Guarda el informe en la carpeta `bi` del proyecto con este nombre:

   `Panel_Rotacion_Empleados.pbix`

3. Conserva el libro Excel junto al PBIX para que se pueda actualizar el informe.

## 5. Comprobar las cifras

La tabla de modelos debe mostrar, redondeado a tres decimales:

| Modelo | ROC-AUC | PR-AUC | F1 evaluado |
|---|---:|---:|---:|
| Regresión Logística | 0,823 | 0,575 | 0,542 |
| Random Forest | 0,803 | 0,540 | 0,523 |
| MLP | 0,814 | 0,559 | 0,522 |

Las tasas de departamento y horas extra son **observadas**. Los recuentos de `Senales_OOF_LogReg` son señales de evaluación fuera de muestra y no tasas observadas.

