# Proyecto 1 — Monitoreo transaccional: detectar lo que el orden revela

Banco del Altiplano (ficticio). Ruta A: datos sintéticos con generador propio. Equipo: Luis Pedro
Lira y Mario Rocha.

## Reproducción

Orden de ejecución (cada notebook consume lo que produce el anterior):

1. `generador.ipynb` → genera `data/transacciones.parquet` (semilla 42, reproducible: el propio
   notebook valida que dos corridas con la misma semilla producen el mismo dataset fila por fila).
2. `linea_base.ipynb` → variables agregadas, partición temporal 70/15/15, Modelo A (boosting de
   árboles), función de evaluación común y análisis económico. Guarda `artefactos/modelo_a_boosting.joblib`
   y `artefactos/parametros_modelo_a.json`.
3. `modelo_secuencial.ipynb` → secuencias por cliente, Modelo B (GRU/LSTM, elegido por validación),
   apuesta C (modelo híbrido), las dos pruebas de falsificación y el análisis económico final A/B/C.
   Guarda `artefactos/modelo_b_secuencial.keras` y `artefactos/parametros_modelo_b.json`.
4. `proyecto1_lira_rocha.ipynb` — fusión de los tres anteriores en un solo notebook ejecutado, el
   entregable final.

### Resultados clave (conjunto de prueba, una sola mirada)

| | AUC-PR | Precisión | Exhaustividad | F1 | Umbral | Ahorro mensual estimado |
|---|---|---|---|---|---|---|
| Modelo A (agregados) | 0.847 | 0.441 | 0.917 | 0.595 | 0.05 | Q2,877,267 |
| Modelo B (secuencial, GRU) | 0.924 | 0.483 | 0.940 | 0.638 | 0.40 | Q2,974,533 |
| Modelo C (híbrido, no recomendado) | 0.890 | 0.528 | 0.930 | 0.674 | 0.68 | Q2,966,667 |

Prueba 1 (permutación controlada, Modelo B en test): AUC-PR cae de **0.924 a 0.070** al barajar el
orden dentro de cada secuencia — la caída es de casi el 92%, evidencia central de que el modelo usa
el orden. Apuesta C: AUC-PR de validación A=0.836, B=0.926, C=0.900 → **no superó** el margen de 0.01
declarado de antemano, veredicto **no útil** (tampoco es más barata que B). El Modelo B gana a A y a
C tanto en AUC-PR como en costo mensual — es el candidato recomendado.

`utils_comunes.py` centraliza `construir_features`, `particionar_temporal`, `evaluar_modelo` y
`resumen_economico` para que ambos modelos usen exactamente el mismo preprocesamiento, la misma
partición y la misma métrica — no reimplementaciones separadas que puedan divergir.

### Versiones

- Python 3.12
- pandas 2.3, numpy 2.3, scikit-learn 1.8
- tensorflow / keras 2.21 / 3.15 (CPU; no se usó GPU)
- pyarrow (lectura/escritura de parquet)

`data/` y los pesos entrenados no se versionan en git por tamaño; se regeneran corriendo los
notebooks en orden con la semilla fija.

## Declaración de uso de IA

Se usó Claude Code (Anthropic) como asistente para: (a) diseñar y escribir el pipeline de
construcción de secuencias, la arquitectura del Modelo B y del modelo híbrido C, las dos pruebas de
falsificación y el análisis económico final; (b) redactar la narrativa de `modelo_secuencial.ipynb`
y este README.

Qué se verificó: cada notebook se ejecutó de principio a fin (no solo revisado) y sus salidas
(métricas, tablas, gráficas) son las que produjo esa ejecución, no números inventados. Se revisó
específicamente que la partición temporal no tuviera solapamiento, que el escalamiento de variables
se ajustara solo con `train`, que el umbral de cada modelo se eligiera en `validación` y se aplicara
una sola vez a `test`, y que la apuesta C se evaluara contra el criterio de éxito declarado *antes*
de entrenarla (no se relajó el margen al ver que no lo alcanzaba).

## Tres decisiones técnicas importantes

**1. Qué le da de comer al Modelo B: eventos crudos, no las variables agregadas de A.**
Alternativa considerada: alimentar a B con las mismas variables agregadas causales que a A, más el
orden. Se descartó porque habría contaminado la comparación — B habría podido "ganarle" a A por
tener acceso a la misma información *más* el orden, no por el orden en sí mismo. Evidencia que
inclinó la decisión: con eventos crudos, la Prueba 1 (permutación) muestra una caída grande del
AUC-PR al barajar el orden — si B hubiera visto los agregados, esa caída habría sido más difícil de
interpretar (¿cae por el orden, o porque se perdió algo de los agregados al permutar?).

**2. Arquitectura del Modelo B: GRU, elegida por comparación en validación, no por preferencia previa.**
Alternativa considerada: LSTM. En el filtro inicial (4 épocas, AUC-PR de validación) GRU quedó en
0.907 contra 0.905 de LSTM — prácticamente empatadas (ver `artefactos/parametros_modelo_b.json`,
campo `comparacion_arquitectura_val_auc_pr`). Evidencia que inclinó la decisión: con un resultado tan
cercano, se usó como desempate que una GRU tiene ~25% menos parámetros que una LSTM del mismo tamaño
de estado oculto (3 compuertas contra 4) — mismo desempeño, menor costo de entrenamiento. La GRU
elegida terminó su entrenamiento completo con AUC-PR de validación 0.926.

**3. Umbral de decisión: minimizar costo económico en validación, no maximizar F1 ni exactitud.**
Alternativa considerada: elegir el umbral que maximiza F1, o el estándar 0.5. Se descartó porque
ninguno de los dos refleja que un fraude no detectado (Q4,200) pesa ~23 veces más que bloquear una
transacción legítima (Q180) — un umbral "balanceado" deja pasar fraude que sale mucho más caro que
los falsos positivos que evita. Evidencia que inclinó la decisión: el barrido de costo sobre
validación (`resumen_economico` en `utils_comunes.py`), aplicado igual a A, B y C.

## Candidato al Proyecto Final

- **Modelo conservado:** Modelo B (secuencial, GRU). Artefacto: `artefactos/modelo_b_secuencial.keras`
  + `artefactos/parametros_modelo_b.json` (umbral, estadísticas de escalamiento ajustadas solo con
  train, mapeo de categorías/canal a índices de embedding — todo lo necesario para reproducir sus
  puntajes sin reentrenar). Se descarta el modelo híbrido C como candidato: no superó el margen que
  el propio equipo fijó de antemano para declararlo útil, así que no hay evidencia que justifique su
  complejidad adicional sobre B.
- **Quién usaría el puntaje y qué decisión tomaría:** el área de riesgos/fraude del banco, para
  decidir en tiempo casi real si retener o dejar pasar una transacción entrante. Puntaje ≥ umbral →
  retener para revisión (o rechazar, según el canal); puntaje < umbral → dejar pasar.
- **Contrato preliminar de entrada/salida:**
  - *Entrada:* las últimas ≤10 transacciones de la tarjeta (incluida la actual), cada una con monto,
    categoría de comercio, canal, hora del día, día de la semana y segundos desde la transacción
    anterior. Sin historial suficiente, se rellena por la izquierda (el modelo lo maneja con una
    máscara, no hay que "inventar" transacciones).
  - *Salida:* un puntaje continuo en [0, 1] (probabilidad de fraude). La decisión de umbral es una
    capa aparte, ya calculada en `parametros_modelo_b.json`, para que se pueda ajustar sin reentrenar
    si cambia el costo relativo de FN/FP.
- **Límites, riesgos y datos que faltarían:**
  - Los datos son sintéticos; el desempeño reportado no se traslada automáticamente a fraude real —
    el patrón de "sondeo y fuga" y los otros dos tipos son plausibles pero no están validados contra
    casos reales del banco.
  - El punto de comparación económico es "no interceptar nada", no el motor de reglas real — no se
    contó con sus predicciones históricas para una comparación directa. Antes de producción, haría
    falta ese contraste.
  - El modelo asume que la ventana de las últimas ≤10 transacciones está disponible en tiempo real al
    momento de decidir — depende de la latencia de la infraestructura de eventos del banco, no
    evaluada aquí.
  - No se probó bajo un mecanismo de fraude no visto durante el entrenamiento (fuera del alcance de
    este proyecto); es un riesgo conocido y explícito, no resuelto.
  - Aún no hay API, monitoreo en producción ni reentrenamiento programado — corresponde al Proyecto
    Final, según el enunciado.

## Matriz de evidencias

| Evidencia | Dónde aparece | Conclusión | Limitación |
|---|---|---|---|
| 1. Integridad de datos | `generador.ipynb` (validación de reproducibilidad, tabla de tipos de fraude) | 3 tipos de fraude documentados, ~1.2% tasa de fraude, reproducible con semilla 42 | Datos sintéticos; el generador decide qué tan difícil es cada patrón |
| 2. Comparación común A vs B | `modelo_secuencial.ipynb` (tabla de métricas y curva PR) | B supera a A en AUC-PR, precisión, exhaustividad y F1 sobre la misma partición | Ambos evaluados en un solo horizonte de predicción (por transacción) |
| 3. Valor del orden | `modelo_secuencial.ipynb` (Prueba 1: permutación; Prueba 2: por tipo de fraude) | El AUC-PR de B cae fuertemente al barajar el orden; el desglose por tipo es consistente con esa lectura | La Prueba 2 está confundida por umbrales distintos entre A y B; se usó el puntaje promedio, no solo la exhaustividad |
| 4. Apuesta del equipo | `modelo_secuencial.ipynb` (hipótesis declarada, entrenamiento y veredicto de C) | El híbrido no superó el margen declarado de antemano → veredicto no útil | Un solo entrenamiento por modelo; no se corrieron semillas múltiples para descartar varianza |
| 5. Decisión económica | `linea_base.ipynb` (A) y `modelo_secuencial.ipynb` (B, C, tabla final) | Umbral óptimo por costo en validación para cada modelo; B ahorra más que A al mes | Comparación contra "no interceptar nada", no contra el sistema de reglas real del banco |
| 6. Recomendación y límites | Este README (sección "Candidato al Proyecto Final") | Se recomienda conservar el Modelo B, no el híbrido | Ver límites arriba: datos sintéticos, sin contraste con el sistema real, sin prueba ante fraude no visto |
