# Material para la memoria del TFM
### Modelo de Expected Goals (xG) con Machine Learning

> Documento de apoyo para la redacción de la memoria. Contiene la descripción
> metodológica, el fundamento matemático, el pseudocódigo de los algoritmos
> centrales, la interpretación de resultados, las limitaciones y el trabajo
> futuro, redactados en estilo formal y listos para integrar (con adaptaciones)
> en los capítulos correspondientes.

---

## 1. Planteamiento y objetivos

El *Expected Goals* (xG) cuantifica la probabilidad de que un disparo se
convierta en gol dadas las circunstancias en que se produce. Formalmente, para
un disparo descrito por un vector de características `x`, el xG es la
probabilidad condicional

```
xG(x) = P(gol = 1 | X = x).
```

El objetivo del trabajo es estimar esta función de probabilidad mediante
aprendizaje automático y, sobre todo, hacerlo de forma **calibrada**: que cuando
el modelo asigna xG = 0.3 a un conjunto de disparos, aproximadamente el 30 % de
ellos terminen efectivamente en gol. Esta exigencia distingue el problema de una
clasificación binaria al uso, donde basta con ordenar correctamente los casos.

Se persiguen tres objetivos específicos:

1. Construir un modelo de xG con buena resolución y calibración, comparando
   familias de modelos bajo métricas propias de la estimación probabilística.
2. Validar la generalización del modelo más allá de la muestra de entrenamiento,
   incluyendo la generalización a competiciones no vistas.
3. Aplicar el modelo al análisis de rendimiento en finalización, distinguiendo
   la habilidad real de la variación aleatoria mediante inferencia estadística
   clásica y bayesiana.

---

## 2. Datos y unidad de análisis

La unidad de análisis es el **disparo**. Cada disparo se representa por una fila
con su localización en el campo, su contexto temporal y táctico, e información
sobre la disposición defensiva en el instante del tiro. La fuente primaria es
StatsBomb Open Data, que proporciona, además de las coordenadas, los
*freeze frames*: la posición de todos los jugadores en el momento del disparo.

Para garantizar la reproducibilidad sin depender de la disponibilidad de red, el
proyecto incorpora un **generador de datos sintéticos** cuyo proceso generador es
un modelo logístico conocido sobre la geometría del disparo. Este generador
define un xG verdadero `xG*(x)` —un *oráculo*— contra el cual es posible medir
directamente el error de estimación del modelo, algo imposible con datos reales,
donde solo se observa el resultado binario y nunca la probabilidad subyacente.

Los penaltis se excluyen del modelado: su probabilidad de conversión es
aproximadamente constante (~0.76) e independiente de las características
geométricas, por lo que distorsionarían el aprendizaje de la relación
geometría→probabilidad.

---

## 3. Ingeniería de variables

### 3.1 Geometría del disparo

Las dos variables físicas fundamentales se derivan de la posición del disparo
respecto a la portería. Con la geometría de campo de StatsBomb (120 × 80), la
portería se sitúa en `x = 120` entre los postes `y = 36` e `y = 44`.

**Distancia al centro de la portería.** Para un disparo en `(x, y)`:

```
d = sqrt( (120 - x)^2 + (40 - y)^2 ).
```

**Ángulo de visión de la portería.** El ángulo subtendido por los dos postes
desde la posición del disparo se obtiene por la ley del coseno. Si `a` y `b` son
las distancias del disparo a cada poste y `w = 8` es la anchura de la portería:

```
theta = arccos( (a^2 + b^2 - w^2) / (2ab) ).
```

Este ángulo es la variable geométrica más informativa: captura simultáneamente
lo cerca y lo centrado que está el disparo. Un disparo lejano y escorado
subtiende un ángulo pequeño; un disparo cercano y centrado, un ángulo grande.

### 3.2 Transformaciones no lineales

La relación entre geometría y probabilidad de gol es marcadamente no lineal. Se
añaden transformaciones que la hacen accesible a modelos lineales: el logaritmo
de la distancia, su inversa, y la interacción ángulo × distancia. Para el modelo
logístico con B-splines, estas transformaciones se sustituyen por bases de
*splines* que aproximan cualquier dependencia suave (ver §4.2).

### 3.3 Contexto táctico y estado del partido

Se incorporan variables de contexto: parte del cuerpo del disparo, tipo de
jugada, presencia de regate previo, y el **estado del partido** (`game_state`):
si el equipo va ganando, empatando o perdiendo en el momento del disparo. Esta
última requiere cuidado temporal: el marcador debe reflejar **solo los goles
anteriores** al disparo. Se calcula mediante sumas acumuladas ordenando los
eventos por su índice cronológico real dentro del partido (el `event_index` de
StatsBomb), con un desempate estable para eventos simultáneos, evitando así
cualquier fuga de información del futuro hacia el presente.

---

## 4. Modelos

### 4.1 Familias consideradas

Se comparan seis configuraciones que cubren el espectro interpretabilidad–
flexibilidad: regresión logística (baseline lineal), regresión logística con
B-splines, *random forest*, *gradient boosting* basado en histogramas
(HistGradientBoosting), y variantes. Todos se encapsulan en *pipelines* de
scikit-learn que integran imputación, escalado y codificación, garantizando que
ninguna transformación se ajuste con información del conjunto de validación
(ausencia de fuga en validación cruzada).

### 4.2 La regresión logística con B-splines

Un modelo lineal en las variables originales es un rival débil; uno con la no
linealidad adecuada es un rival serio. Se expande cada variable continua en una
base de B-splines `{B_1(x), ..., B_K(x)}`, de modo que el predictor lineal se
convierte en

```
logit( xG ) = beta_0 + sum_k beta_k B_k(distancia) + sum_l gamma_l B_l(angulo) + ...
```

Esto permite a la regresión logística aproximar la dependencia suave
geometría→probabilidad sin abandonar el marco lineal generalizado. Su inclusión
es deliberada: si el *gradient boosting* solo superase a una logística *sin*
splines, la conclusión "los árboles ganan" sería un hombre de paja. Frente a la
logística con splines, la comparación es justa.

### 4.3 Calibración isotónica

La calibración corrige la discrepancia entre las probabilidades predichas y las
frecuencias observadas. Se emplea regresión isotónica: se ajusta una función
monótona no decreciente `g` que minimiza

```
sum_i ( g(p_i) - y_i )^2     sujeto a   g monotona,
```

donde `p_i` es la probabilidad predicha y `y_i` el resultado. La monotonía
preserva el orden de los disparos (no altera el AUC) mientras ajusta los niveles
de probabilidad (mejora el Brier y el ECE). La calibración se ajusta con
validación cruzada interna sobre el conjunto de entrenamiento, sin tocar el test.

---

## 5. Evaluación y validación

### 5.1 Métricas

Por las razones expuestas, la métrica principal **no** es el ROC-AUC sino el
**Brier score**, el error cuadrático medio de la probabilidad:

```
Brier = (1/n) sum_i ( p_i - y_i )^2.
```

Es una *regla de puntuación propia*: se minimiza en esperanza cuando `p_i` es la
probabilidad verdadera, por lo que premia simultáneamente resolución y
calibración. Se complementa con el **Log Loss** (otra regla propia) y con el
**Expected Calibration Error (ECE)**, que mide directamente la calibración:

```
ECE = sum_{b=1}^{B} (n_b / n) | acc(b) - conf(b) |,
```

donde los disparos se agrupan en `B` tramos por probabilidad predicha, `conf(b)`
es la confianza media del tramo y `acc(b)` la frecuencia real de gol en él.

### 5.2 Particiones: holdout por competición y LOCO

Además de una partición estándar, se evalúa la generalización a una competición
**no vista**: se reserva un torneo completo como test. Se complementa con
validación *Leave-One-Competition-Out* (LOCO): se entrena sobre todas las
competiciones menos una, rotando la reservada, y se reporta la media y la
desviación típica de la métrica entre torneos. Esto mide la robustez del modelo
ante el cambio de distribución entre competiciones.

### 5.3 Cuantificación de la incertidumbre

Las métricas en test se acompañan de intervalos de confianza por *bootstrap*
(2000 remuestreos con reemplazo de los disparos del test). La comparación entre
el mejor modelo y el baseline se realiza con un *bootstrap pareado*: en cada
remuestreo se calcula la diferencia de métrica sobre los mismos disparos, lo que
controla la correlación entre modelos y produce un p-valor empírico de la
diferencia.

### 5.4 Scoring honesto para el análisis aplicado

El análisis por jugador y equipo requiere puntuar **todos** los disparos del
conjunto de datos. Hacerlo con un único modelo entrenado y luego re-puntuar sus
propios datos de entrenamiento introduciría un sesgo de optimismo
(re-sustitución): el xG de esos disparos estaría sobreajustado y la suma de xG se
aproximaría artificialmente a los goles, comprimiendo la dispersión de
(goles − xG) que precisamente se desea medir. Para evitarlo, el xG de cada
disparo se estima por validación cruzada **out-of-fold**: el dataset se divide en
pliegues y cada disparo se puntúa con el modelo entrenado en los pliegues
restantes. Todas las predicciones del análisis son, así, fuera de muestra.

---

## 6. Análisis de finalización: frecuentista y bayesiano

### 6.1 El contraste Poisson-binomial

Bajo la hipótesis nula de que la habilidad finalizadora de un jugador coincide
con su xG, el número de goles `G` que marca es la suma de Bernoulli
independientes con probabilidades `p_1, ..., p_n` (los xG de sus disparos), es
decir una distribución **Poisson-binomial** con

```
mu = sum_i p_i,        sigma^2 = sum_i p_i (1 - p_i).
```

Con suficientes disparos se usa la aproximación normal para construir el
estadístico `z = (G - mu) / sigma` y un p-valor bilateral. Un jugador se declara
sobre- o infra-rendidor solo si su intervalo de confianza para `G - mu` excluye
el cero. Esto evita presentar como hallazgo lo que es variación muestral.

### 6.2 La limitación del enfoque frecuentista

El contraste evalúa a cada jugador de forma aislada. Con cientos de jugadores y
pocos disparos cada uno, dos problemas emergen: las estimaciones individuales son
inestables (un par de goles afortunados disparan el estadístico), y al testear
muchos jugadores a un nivel `alpha` se esperan numerosos falsos positivos por
azar. Empíricamente, sobre datos sintéticos sin habilidad de finalización real,
el contraste marca varios jugadores como significativos: son falsos hallazgos.

### 6.3 El modelo jerárquico bayesiano

El remedio canónico al problema de estimar muchas medias relacionadas es el
modelo jerárquico con *partial pooling* (Efron & Morris, 1975; James–Stein). Se
modela la habilidad de finalización de cada jugador como un efecto aleatorio
`u_i` en la escala log-odds, sobre el xG de sus disparos:

```
g_ij ~ Bernoulli( sigma( eta_ij + u_i ) ),     eta_ij = logit( p_ij ),
u_i  ~ Normal( 0, tau^2 ).
```

Aquí `eta_ij` es un *offset* fijo igual al logit del xG del disparo, de modo que
`u_i` mide exclusivamente la desviación sistemática del jugador respecto del
modelo: `u_i > 0` indica que convierte por encima de su xG, y `exp(u_i)` es el
multiplicador sobre las *odds* de marcar. El hiperparámetro `tau^2` mide la
heterogeneidad real de finalización en la población y **se estima de los datos**.

El efecto clave es el **encogimiento adaptativo**: la estimación de cada jugador
se contrae hacia la media poblacional en proporción a su incertidumbre. Los
jugadores con pocos disparos (señal débil) se encogen mucho; los de muchos
disparos (señal fuerte) conservan su estimación. El grado de encogimiento lo
decide `tau`, no el analista.

### 6.4 Inferencia por Bayes empírico (EM + Laplace)

Se estima `tau^2` por máxima verosimilitud marginal con un algoritmo EM en el
que la marginal por jugador se aproxima por Laplace. La log-posterior individual
de `u` (con `tau^2` fijo) es cóncava:

```
ell(u)   = sum_j [ g_ij (eta_ij + u) - log(1 + e^{eta_ij + u}) ] - u^2 / (2 tau^2),
ell'(u)  = sum_j [ g_ij - sigma(eta_ij + u) ] - u / tau^2,
ell''(u) = - sum_j sigma(eta_ij+u)(1 - sigma(eta_ij+u)) - 1 / tau^2.
```

- **Paso E.** Para `tau^2` fijo, se halla el MAP `u_hat_i` por Newton-Raphson
  (convergencia cuadrática por la concavidad) y su varianza posterior por Laplace
  `s_i^2 = 1 / (-ell''(u_hat_i))`.
- **Paso M.** Se actualiza la varianza poblacional con el segundo momento
  posterior agregado: `tau^2 <- (1/N) sum_i ( u_hat_i^2 + s_i^2 )`.

Se itera hasta la convergencia de `tau`. El término `s_i^2` en el paso M hace que
`tau` descienda con convergencia sublineal cuando no hay heterogeneidad real, por
lo que el criterio de parada combina cambio relativo y absoluto.

La probabilidad posterior de que un jugador finalice por encima de su xG se
aproxima por `P(u_i > 0) = Phi( u_hat_i / s_i )`. Un jugador se considera
finalizador excepcional solo si esta probabilidad supera un umbral alto (p. ej.
0.95).

---

## 7. Pseudocódigo de los algoritmos centrales

### 7.1 Scoring out-of-fold honesto

```
ENTRADA: dataset de disparos X con resultado y; modelo M; nº de pliegues k
SALIDA:  xG_oof[i] para cada disparo i (estimado fuera de muestra)

dividir X en k pliegues estratificados {F_1, ..., F_k}
para cada pliegue F_j:
    entrenar copia de M (opcionalmente calibrada) sobre X \ F_j
    para cada disparo i en F_j:
        xG_oof[i] <- modelo_entrenado.predecir_probabilidad(X[i])
devolver xG_oof
```

### 7.2 Ajuste jerárquico bayesiano (EM + Laplace)

```
ENTRADA: por jugador i, log-odds base eta_ij y resultados g_ij
SALIDA:  habilidades u_hat_i, errores s_i, y tau

tau2 <- tau_inicial^2
repetir hasta convergencia de tau:
    suma <- 0
    para cada jugador i:
        u_hat_i <- argmax_u ell(u; eta_i, g_i, tau2)     # Newton-Raphson
        s2_i    <- 1 / (-ell''(u_hat_i))                 # Laplace
        suma    <- suma + u_hat_i^2 + s2_i
    tau2 <- suma / N                                     # paso M
para cada jugador i:
    prob_better_i <- Phi( u_hat_i / s_i )
devolver (u_hat, s, sqrt(tau2), prob_better)
```

---

## 8. Interpretación de resultados

### 8.1 Dataset final y protocolo experimental

La ejecución final del proyecto se realiza con datos reales de StatsBomb Open
Data. El conjunto contiene 5.829 disparos brutos procedentes de cuatro torneos
internacionales: FIFA World Cup 2018, UEFA Euro 2020, FIFA World Cup 2022 y UEFA
Euro 2024. Tras excluir los penaltis, el modelado se realiza sobre 5.606
disparos no penales y 507 goles, con una tasa media de gol del 9,04 %. La UEFA
Euro 2024 se reserva como competición no vista para el test externo, lo que
permite evaluar la generalización entre torneos y evita una partición aleatoria
excesivamente optimista.

### 8.2 Comparación de modelos

En el test externo de UEFA Euro 2024, el modelo HistGradientBoosting calibrado
obtiene un ROC-AUC de 0,7444, un PR-AUC de 0,2032, un Brier Score de 0,0659, un
Log Loss de 0,2406 y un ECE de 0,0176. La versión sin calibrar del mismo modelo
alcanza un Brier de 0,0665, un Log Loss de 0,2424 y un ECE de 0,0206. Por tanto,
la calibración isotónica introduce una mejora ligera en la calidad probabilística
y en la calibración, aunque no debe interpretarse como una mejora drástica.

La regresión logística y la regresión logística con B-splines obtienen valores de
ROC-AUC y PR-AUC competitivos, pero presentan un problema grave de calibración:
sus probabilidades medias predichas se sitúan alrededor de 0,36--0,37 frente a
una tasa real de gol del 0,075 en el test. En consecuencia, aunque ordenan
razonablemente los disparos, no son adecuadas como modelo final de xG, ya que el
xG requiere probabilidades interpretables y agregables. Este resultado refuerza
la elección del HistGradientBoosting calibrado como modelo principal.

### 8.3 Incertidumbre estadística y comparación pareada

Los intervalos de confianza por bootstrap muestran que la estimación del modelo
principal tiene una incertidumbre no despreciable, especialmente en ROC-AUC y
PR-AUC, dado que el test contiene 98 goles sobre 1.304 disparos. Para el modelo
calibrado, el IC 95 % del ROC-AUC es [0,6926; 0,7939], el del PR-AUC es
[0,1511; 0,2771], el del Brier Score es [0,0560; 0,0768] y el del Log Loss es
[0,2110; 0,2729].

La comparación pareada frente a la regresión logística confirma que las
diferencias en Brier Score y Log Loss son estadísticamente claras a favor del
modelo calibrado, mientras que la diferencia en ROC-AUC no resulta significativa.
Esto es coherente con la tesis metodológica del trabajo: para xG, la calidad de
las probabilidades debe tener prioridad sobre la mera discriminación.

### 8.4 Validación LOCO y benchmark externo

La validación Leave-One-Competition-Out muestra un rendimiento medio de ROC-AUC
0,7680, PR-AUC 0,3002, Brier 0,0730 y ECE 0,0153. La UEFA Euro 2024 es el torneo
más exigente en términos de ROC-AUC y PR-AUC, lo que sugiere cierto cambio de
distribución entre competiciones, pero el Brier y el ECE permanecen en rangos
razonables.

La comparación frente al xG oficial de StatsBomb aporta una validación externa
del modelo. Sobre 5.606 disparos no penales, la correlación de Pearson entre el
xG propio y el xG de StatsBomb es 0,8268 y la de Spearman es 0,8041. La suma de
xG propio es 503,73, muy próxima a los 507 goles reales, mientras que StatsBomb
acumula 519,44. Sin embargo, StatsBomb obtiene un Brier Score inferior
(0,0679 frente a 0,0719), por lo que el resultado debe interpretarse como una
aproximación académica coherente a una referencia profesional, no como una
superación del modelo de StatsBomb.

### 8.5 Finalización: rankings, significación y encogimiento bayesiano

Los rankings simples de goles menos xG son útiles para explorar el rendimiento
ofensivo, pero pueden inducir a sobreinterpretación, especialmente en jugadores
con pocos disparos. Por ello, el proyecto complementa el ranking crudo con un
contraste Poisson-binomial y con un modelo jerárquico bayesiano. Además, la tabla
frecuentista incorpora una corrección Benjamini-Hochberg para controlar la tasa
de falsos descubrimientos al evaluar simultáneamente a muchos jugadores.

El análisis bayesiano muestra un patrón prudente: aunque algunos jugadores
aparecen con sobre-rendimientos brutos elevados, el modelo jerárquico regulariza
estas diferencias mediante *partial pooling*. En la ejecución final, ningún
jugador supera una probabilidad posterior de 0,95 de finalizar sistemáticamente
por encima de su xG; el valor máximo se sitúa alrededor de 0,925. Esto indica que
los rankings deben leerse como evidencia exploratoria y no como prueba concluyente
de habilidad finalizadora extraordinaria.

## 9. Limitaciones

1. **Información defensiva parcial.** Aunque se cuentan los defensores en el cono
   de tiro, no se modela la presión, la trayectoria del balón ni la posición del
   portero con detalle. Un *post-shot* xG (que incorpora la localización del
   remate dentro de la portería) escapa al alcance del modelo pre-disparo.
2. **Aproximación normal en el contraste frecuentista.** El test
   Poisson-binomial usa la aproximación normal, cuya validez decae con muy pocos
   disparos. Se añade corrección FDR para comparaciones múltiples, pero la
   interpretación principal debe apoyarse en el modelo bayesiano jerárquico.
3. **Laplace en la inferencia bayesiana.** La aproximación de Laplace de la
   marginal es exacta solo asintóticamente; para inferencia plena del posterior
   (R-hat, ESS, intervalos HDI) se proporciona un backend MCMC (PyMC) opcional.
4. **Dependencia de la calidad del etiquetado.** El modelo hereda cualquier
   sesgo o ruido del etiquetado de eventos de la fuente de datos.

---

## 10. Trabajo futuro

1. **Post-shot xG (xGOT).** Modelar la probabilidad de gol condicionada también a
   la localización del remate dentro del marco, separando la calidad de la
   ocasión de la calidad del remate.
2. **Modelo bayesiano completo por MCMC.** Sustituir la aproximación EM+Laplace
   por muestreo NUTS (PyMC), obteniendo intervalos de credibilidad exactos y
   diagnósticos de convergencia, y extender a un modelo de dos niveles
   (jugador anidado en equipo).
3. **xThreat y valor de acciones.** Integrar el xG en un marco de valoración de
   posesiones (*expected threat*) para medir la contribución de acciones previas
   al disparo.
4. **Variables espaciotemporales ricas.** Incorporar el *tracking* completo
   (velocidades, presión, líneas defensivas) mediante redes neuronales sobre los
   *freeze frames*, comparándolas con el modelo tabular como baseline.
5. **Calibración dependiente del contexto.** Estudiar si la calibración varía por
   competición o por estado del partido y, en su caso, calibrar por estratos.

---

## Referencias orientativas

- Efron, B. & Morris, C. (1975). *Data analysis using Stein's estimator and its
  generalizations*. JASA.
- Gelman, A. et al. *Bayesian Data Analysis* (3.ª ed.), caps. sobre modelos
  jerárquicos.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good probabilities with
  supervised learning*. ICML (calibración).
- Documentación de StatsBomb Open Data.
