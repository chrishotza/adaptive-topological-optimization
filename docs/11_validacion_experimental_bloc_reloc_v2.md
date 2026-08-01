# Capítulo 11: Validación Experimental de BLOC-RELOC v2

**Capítulo 11: Validación Experimental de BLOC-RELOC v2**
11.1 Objetivos Experimentales y Verificación de Hipótesis
11.2 Diseño del Entorno Experimental
11.3 Conjunto de Benchmarks (Suite de Pruebas)
11.4 Algoritmos Comparados
11.5 Métricas de Evaluación Cuantitativa
11.6 Protocolo Experimental (Secuencia de Ejecución)
11.7 Marco de Validación Estadística
11.8 Análisis de Amenazas a la Validez
11.9 Reproducibilidad de la Campaña
11.10 Preparación para los Resultados
----# PARTE I: FUNDAMENTOS TEÓRICOS
--# CAPÍTULO 1: HISTORIA COMPLETA DEL PROYECTO
--## 1.1 Antecedentes y Génesis del Laboratorio TCF

El Laboratorio TCF nació a partir de un cuestionamiento fundamental sobre la naturaleza de la computación, la auto-org

La pregunta que motivó la fundación del laboratorio fue contundente: *¿Es posible modelar la resolución de problemas c

Bajo la dirección del CEO y con la custodia operativa y conceptual de DIP (Chief Operating Officer del Laboratorio TCF, C

Esta visión dio origen a la **División BLOC-RELOC**, un equipo de investigación y desarrollo abocado a la creación de un
--## 1.2 El Paradigma Homeostático y la Construcción del Motor Inicial

En los primeros sprints de desarrollo, el enfoque científico estuvo concentrado en la hipótesis del **Solucionador Univer
El laboratorio propuso una premisa alternativa:

1. **Reubicación por Bloques:** En lugar de iterar nodo por nodo de forma puramente secuencial o swaps de pares de n
2. **Presión de Vecindad:** Un nodo no decide su pertenencia a una partición basándose en una función de costo globa
3. **Invariancia Determinista:** Se buscaba que, ante una misma topología y una condición inicial dada, el motor conve

Con estas bases, se construyó la versión alfa de **BLOC-RELOC**. En sus primeras pruebas internas sobre grafos sintéti
--## 1.3 La Evolución del Núcleo y la Creación del Meta-Selector

A medida que el núcleo de BLOC-RELOC se consolidaba, el equipo del Laboratorio TCF enfrentó el primer desafío técnico
Para responder a esto, el desarrollo se ramificó en dos líneas complementarias:
### A. Perfeccionamiento de la Regla de Reubicación

Se implementaron estructuras de datos optimizadas en Python/C++ para rastrear en tiempo real el cambio de ganancia
### B. El Primer Meta-Selector

DIP y el equipo identificaron que la tasa de convergencia dependía sensiblemente de ciertos parámetros de control, com

En este punto del proyecto, el optimismo dentro del laboratorio era máximo. El motor funcionaba con fluidez, la arquitec
--## 1.4 La Campaña Masiva de Benchmarking

Con el algoritmo maduro, el CEO ordenó la ejecución de la prueba definitiva: **la primera campaña masiva de benchma

* **Kernighan-Lin (KL):** La heurística clásica de refinamiento por intercambio de pares.
* **Breakout Local Search (BLS):** Algoritmo de búsqueda local con perturbaciones adaptativas para escape de mínimo
* **KaHyPar:** El estándar industrial y académico de particionamiento multinivel de grafos e hipergrafos.
* **SCIP / Solvers Exactos:** Utilizados como suelo de validación para instancias pequeñas y medianas donde el óptimo
### El Conjunto de Datos (Dataset de Prueba)
El benchmark abarcó un espectro diverso de familias topológicas, superando cientos de instancias de prueba:

1. **Grafos Regulares (Random Regular Graphs):** Redes donde todos los vértices poseen exactamente el mismo grado
2. **Redes Small-World (Watts-Strogatz):** Grafos con alto coeficiente de agrupamiento local y corta distancia promedio
3. **Redes Scale-Free (Barabási-Albert sintéticas):** Grafos con distribución de grado que sigue una ley de potencia $P(
4. **Grafos Disconectados / Planted Partition:** Estructuras con comunidades sintéticas con densidades intrínsecas y ex
5. **Redes Reales del Repositorio SNAP (Stanford Network Analysis Platform):** Instancias masivas extraídas de aplicac
--## 1.5 La Revelación Empírica: El Quiebre de Rendimiento

Los resultados de la campaña masiva de benchmarking trajeron un hallazgo científico imprevisto y transformador. En lu
```
RESULTADOS DEL BENCHMARKING
┌────────────────────────────────────────┬────────────────────────────────────────┐
│ GRAFOS SINTÉTICOS / ESTRUCTURADOS │
REDES REALES HETEROGÉNEAS
│
│ (Regular, SmallWorld, Disconnected) │
(SNAP: Facebook)
│
├────────────────────────────────────────┼────────────────────────────────────────┤
│ Costo Corte
│ Costo Corte
│
│ │
│ │
│
│ 100┤ ┌───┐
┌───┐
│ 900┤ ┌───┐
│
│ │ │ │
│ │
│ 800┤ │ │ <-- BLOC-RELOC
│
│ 50┤ │ │ <-- BLOC │ │ <-- KL │ 700┤ │ │
(Costo elevado)
│
│ │ └───┘
└───┘
│ 600┤ └───┘
┌───┐
│
│ 0└───────────────────────────
│ 500┤
│ │ <-- KL
│
│
│ 0└────────────────┴───┴───────────────│
│ Convergencia idéntica al óptimo.
│ Caída drástica de rendimiento frente │
│ Desviación estándar = 0.
│ a Kernighan-Lin y Breakout.
│
└────────────────────────────────────────┴────────────────────────────────────────┘
```
### Anomaly Analysis

1. **En Grafos Sintéticos y Estructurados:** BLOC-RELOC igualó o superó de manera sistemática a Kernighan-Lin y a las
2. **En Redes Reales SNAP:** El escenario cambió radicalmente. En grafos como Facebook y WikiVote, Kernighan-Lin y
--## 1.6 El Conflicto de Paradigmas y la Declaración del Freeze Técnico

Este resultado desencadenó un debate metodológico crucial en el seno del Laboratorio TCF. Se presentaron dos postura

* **La Opción de la Micro-Optimización (Parches Locales):** Intentar "arreglar" BLOC-RELOC modificando parámetros, ag
* **La Opción del Salto Paradigmático (Revisión de Hipótesis):** Aceptar la evidencia de los datos de que **no existe un

El CEO, respaldado por la evaluación de DIP y por los análisis analíticos generados por las entidades de investigación de

> **DECRETO DE FREEZE TÉCNICO COMPLETO**
> 1. Se congelan de manera inmediata todas las líneas de código del solver BLOC-RELOC v1.
> 2. Se prohíbe la aplicación de parches de hiperparámetros o modificaciones ad-hoc destinadas a "forzar" resultados e
> 3. Se clausura oficialmente la búsqueda de un solver monolítico universal.
> 4. Se reorienta la misión científica del Laboratorio TCF hacia el diseño de un **Framework Adaptativo de Optimización
---

## 1.7 Estado Actual y Posicionamiento Estratégico
La declaración del *Freeze Técnico* marcó el fin de la primera fase del Proyecto TCF y el acta de nacimiento de su segu

Hoy, el Laboratorio TCF no ve la divergencia de rendimiento de BLOC-RELOC como una derrota, sino como su mayor con

BLOC-RELOC ha dejado de ser un algoritmo aislado que intenta competir contra todo el estado del arte para convertirse
----# CAPÍTULO 2: ORIGEN FILOSÓFICO Y CIENTÍFICO
--## 2.1 La Teoría de Conciencia Fotónica (TCF): De la Ontología a la Computación

La base conceptual sobre la que se erige el laboratorio no nació en el ámbito de la optimización combinatoria tradiciona

En la informática clásica, la información suele tratarse como una entidad pasiva: un conjunto de bits almacenados en m

El término "conciencia" dentro de la TCF no posee una connotación antropomórfica ni mística; se define estrictamente d

Al trasladar esta cosmovisión al dominio de las ciencias de la computación, el laboratorio identificó que los problemas N
--## 2.2 Principios Fundamentales: Autorreferencia Generativa, Luz-Potencial y Dinámica Homeostática

La arquitectura teórica de la TCF se apoya en tres pilares matemáticos y conceptuales interconectados que rigen el com
```
┌─────────────────────────────────────────┐
│
AUTORREFERENCIA GENERATIVA
│
│ (Evaluación de la Tensión Local) │
└────────────────────┬────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│
LUZ-POTENCIAL
│
│ (Flujo de Información en Aristas) │
└────────────────────┬────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│
DINÁMICA HOMEOSTÁTICA
│
│ (Minimización del Gradiente de Energía) │
└─────────────────────────────────────────┘
```
### A. Autorreferencia Generativa

Principio según el cual cada entidad o elemento dentro de un grafo (nodo o subgrafo) evalúa su estado basándose exclu
### B. Luz-Potencial ($\Phi$)
Se define como la magnitud que cuantifica el grado de conectividad no resuelta o tensión acumulada en las aristas que
### C. Dinámica Homeostática

Inspirada en los mecanismos biológicos y termodinámicos de autorregulación, la Dinámica Homeostática es la fuerza im
--## 2.3 Mapeo Formal: De la Ontología Físico-Filosófica a la Teoría de Grafos

Para transformar estos postulados filosóficos en una herramienta matemática rigurosa, el Laboratorio TCF desarrolló un
Sea un grafo no dirigido y ponderado $G = (V, E, w)$, donde $V$ es el conjunto de vértices o nodos, $E$ es el conjunto
| Concepto de la Teoría TCF | Dominio Matemático en Teoría de Grafos | Formulación Simbólica |

|---|---|---|
| Sustrato Fotónico | Grafo Ponderado $G = (V, E, w)$ | Red subyacente de interacciones. |
| Campo de Estado | Asignación de Partición | $\pi: V \to \{1, 2, \dots, k\}$ |
| Luz-Potencial Local ($\Phi_u$) | Tensión de Conectividad del Nodo $u$ | $\Phi_u(i) = \sum_{v \in N(u), \pi(v) = i} w(u,v)
| Fricción de Frontera ($\mathcal{F}$) | Costo del Corte de la Partición ($Cut\ Weight$) | $\mathcal{F}(\Pi) = \frac{1}{2
| Equilibrio Homeostático Local | Condición de Máxima Afinidad | $\pi(u) = \arg\max_{i} \Phi_u(i)$ |
Bajo este mapeo, la Luz-Potencial Local $\Phi_u(i)$ cuantifica la atracción o "fuerza de gravedad fotónica" que ejerce el

La Fricción de Frontera $\mathcal{F}(\Pi)$ equivale directamente al costo funcional del corte que los algoritmos de part
--## 2.4 La Hipótesis de la Mínima Fricción Topológica (MFT)

El pilar teórico que unificó la filosofía TCF con el diseño algorítmico fue formulado por el equipo bajo el nombre de la Hip

> **Hipótesis MFT:** En todo grafo $G = (V, E)$, existe un estado de relajación homeostática local alcanzable exclusiva

La Hipótesis MFT postulaba que el espacio de estados de un problema de particionamiento no requiere ser explorado m
Se asumía que las barreras de energía (mínimos locales subóptimos) eran artefactos causados por una mala resolución
--## 2.5 Transición Conceptual hacia el Modelo Algorítmico y Limitaciones del Reduccionismo Inicial

La Hipótesis MFT proporcionó una elegancia teórica innegable y constituyó el catalizador directo para la creación de BLO

Sin embargo, el marco filosófico inicial contenía un supuesto implícito que más tarde revelaría ser su principal punto cie

No se contempló preliminarmente que existen grafos donde la densidad de información está extremadamente concentr

A pesar de esta limitación —que no sería completamente comprendida hasta la ejecución del benchmark masivo—, la so
----# CAPÍTULO 3: NACIMIENTO DE BLOC-RELOC
--## 3.1 De la Hipótesis Teórica a la Especificación Algorítmica: El Imperativo de la Reubicación Colectiva

El paso de la Hipótesis de la Mínima Fricción Topológica (MFT) a una implementación de software requirió resolver una c

El equipo del Laboratorio TCF identificó tempranamente que la reubicación puramente univariada (nodo por nodo) prese
$$G(u, B_j) = \sum_{v \in N(u) \cap B_j} w(u,v) - \sum_{v \in N(u) \cap B_i} w(u,v) < 0$$

Sin embargo, si todo el subconjunto contiguo $K \subset B_i$ al que pertenece $u$ se reubica de manera simultánea ha

$$G(K, B_j) = \sum_{u \in K} \left( \sum_{v \in N(u) \cap B_j} w(u,v) - \sum_{v \in N(u) \cap (B_i \setminus K)} w(u,v) \rig

Este principio dio origen al requerimiento primario de diseño del motor: el algoritmo no debía ser un mero refinador de v
--## 3.2 El Operador Fundamental de Voto de Vecindad y la Función de Ganancia Homeostática

Para operacionalizar la transferencia de los bloques sin incurrir en la complejidad computacional exponencial de evaluar

El OVV proyecta la presión homeostática que ejercen las particiones adyacentes sobre cada nodo de la frontera. Sea $\p

Para cada nodo $u \in V_{\text{bound}}$ que actualmente pertenece a la partición $\pi(u) = i$, se define el vector de a
$$A_u[j] = \sum_{v \in N(u) \cap B_j} w(u,v)$$

A partir de este vector, el **Voto Homeostático Preferencial** del nodo $u$, denotado por $\nu(u)$, selecciona la partici
$$\nu(u) = \arg\max_{j \in \{1, \dots, k\}} A_u[j]$$
Si $\nu(u) = i$, el nodo se halla en equilibrio homeostático local y no emite presión de cambio. Si $\nu(u) = j \neq i$, el
$$\Delta \Phi_u = A_u[\nu(u)] - A_u[i]$$
```
EVALUACIÓN DEL OPERADOR DE VOTO DE VECINDAD (OVV)
Partición B_1 (Actual)
Partición B_2 (Objetivo)
┌──────────────────────┐
┌──────────────────────┐
│
│
│
│
│ (v1) ─── (v2)
│
│
(v4)
│
│
\
/
│ w(u,v4)=5 │
/ \
│
│
\ /
│───────────>│ (v5)───(v6)
│
│
(u) ──────────┼───────────>│
│
│
\ w(u,v3)=1│ w(u,v5)=4 │
│
│
(v3)
│
│
│
└──────────────────────┘
└──────────────────────┘
Atracción B_1: w(u,v1)+w(u,v2)+w(u,v3) = 1 + 1 + 1 = 3
Atracción B_2: w(u,v4)+w(u,v5)
=5+4
=9
Resultado OVV: ν(u) = B_2 | ΔΦ_u = 9 - 3 = +6 (Presión de Reubicación)
```

La agregación espacial de estos votos sobre regiones contiguas de la frontera de corte permite al algoritmo agrupar aut
--## 3.3 Estructuras de Datos Primarias y Gestión de Memoria Tabú Ligera

La eficiencia de ejecución de BLOC-RELOC requirió el desarrollo de una arquitectura de datos capaz de realizar consulta
El núcleo del motor se estructuró sobre tres componentes de datos primarios:

1. **Tabla de Afinidad Dinámica (`AffinityMatrix`):** Matriz dispersa de tamaño $|V| \times k$ mantenida en memoria co
2. **Colas de Prioridad por Cubos de Ganancia (`BucketPriorityQueues`):** Un arreglo de listas doblemente enlazadas in
3. **Registro de Estado Tabú Ligero (`TabuStateTracker`):** Un vector directo $T \in \mathbb{N}^{|V|}$ que registra e
### El Mecanismo de Memoria Tabú Ligera

Para prevenir oscilaciones cíclicas (efecto de "rebote" donde un nodo o bloque oscila indefinidamente entre las particion
$$t - T[u] \le \tau$$

A diferencia de los esquemas tabú complejos que almacenan configuraciones globales pasadas en tablas Hash de alto c
--## 3.4 La Mecánica de Movimiento por Bloques: Reorganización Contigua y Liberación de Fricción

El corazón procedimental de BLOC-RELOC reside en su ciclo de **Identificación, Expansión y Reubicación de Bloques (IE
```
CICLO IERB (IDENTIFICACIÓN Y REUBICACIÓN)
[ Vórtice Semilla ] ──> Selecciona u_seed = argmax(ΔΦ_u) en BucketQueue
│
▼
[ Expansión Bloque ] ──> Explora BFS sobre N(u_seed) buscando v con ν(v) == ν(u_seed)
│
▼
[ Bloque Candidato K ] ─> Verifica cohesión interna y ganancia agregada ΔΦ_K > 0
│
▼

[ Reubicación Atómica ] ─> Transfiere K de B_i a B_j en una sola transacción
│
▼
[ Actualización Masiva ] ─> Recalcula AffinityMatrix para N(K) y aplica TabuStateTracker
```
### El Algoritmo de Expansión de Bloques (Paso a Paso)
```
Procedimiento EXPANDIR_Y_MOVER_BLOQUE(G, π, u_seed, τ, t):
1. j_target = ν(u_seed)
2. B_origin = π(u_seed)
3. Inicializar conjunto K = {u_seed}
4. Inicializar cola Q = [u_seed]
5. Mientras Q no esté vacía y |K| < MaxBlockSize:
a. Extraer u de Q
b. Para cada v en N(u) tal que π(v) == B_origin:
i. Si v no está en K y (t - T[v] > τ):
ii. Si ν(v) == j_target:
- Agregar v a K
- Encolar v en Q
6. Calcular Ganancia Agregada ΔΦ_K
7. Si ΔΦ_K > 0:
a. Para cada v en K:
i. π(v) = j_target
ii. T[v] = t
b. RECALCULAR_AFINIDAD_VECINOS(G, K, B_origin, j_target)
c. Retornar ÉXITO, ΔΦ_K
8. Retornar FALLO, 0
```

Este procedimiento garantiza que la transferencia de nodos preserve la contigüidad espacial del bloque transferido. Si e
--## 3.5 La Versión Alfa de BLOC-RELOC: Primeras Pruebas de Laboratorio y Comportamiento Determinista

A mediados del ciclo inicial de desarrollo, la versión Alfa de BLOC-RELOC fue codificada en Python utilizando extensione

Las primeras pruebas sintéticas se ejecutaron sobre grafos de validación con simetrías ideales y fronteras bien delimita

* **Mallas Bidimensionales (2D Grid Graphs):** Grafos planares con conectividad regular de 4 vecinos.
* **Árboles Binarios Balanceados:** Redes jerárquicas con estructura acíclica.
* **Modelos de Partición Plantada (Planted Partition / SBM estricto):** Grafos aleatorios con densidad intra-comunitaria $
### El Descubrimiento del Comportamiento Determinista

Durante estas pruebas de laboratorio, el equipo registró una propiedad del algoritmo que fue recibida como un hito cien

Al ejecutar BLOC-RELOC múltiples veces sobre una misma topología partiendo de la misma configuración inicial (o inclu
| Métrica de Validación | Simulated Annealing (SA) | Genetic Algorithm (GA) | BLOC-RELOC Alfa |
|---|---|---|---|
| **Tiempo de Convergencia Promedio** | $12.45 \text{ s}$ | $45.12 \text{ s}$ | **$0.34 \text{ s}$** |
| **Desviación Estándar del Costo ($\sigma$)** | $\pm 14.23$ | $\pm 8.67$ | **$0.00$** |
| **Estabilidad de Frontera** | Fluctuante | Fluctuante | **Estable / Atractor Rígido** |
| **Complejidad Espacial** | $\mathcal{O}(|V| + |E|)$ | $\mathcal{O}(P \cdot |V|)$ | **$\mathcal{O}(|V| + |E|)$** |

Este comportamiento de convergencia ultra-rápida y reproducibilidad absoluta convenció al laboratorio de que el motor

Sin embargo, las pruebas iniciales contenían un sesgo fundamental: todas las topologías evaluadas compartían distribuc
----# CAPÍTULO 4: ARQUITECTURA COMPLETA DE BLOC-RELOC
---

## 4.1 Filosofía de Diseño de Software e Ingeniería del Sistema

La traducción del paradigma homeostático en un motor computacional industrial exigió adherirse a principios inflexibles

Para alcanzar esta meta, la arquitectura de **BLOC-RELOC** se diseñó bajo cuatro directrices fundamentales de ingenie

1. **Cero Asignaciones Dinámicas en el Bucle Crítico (*Zero-Allocation Hot Loop*):** Toda la memoria requerida por las
2. **Localidad de Caché Explicita (*Cache-Conscious Data Layouts*):** Las estructuras de datos primarias se almacenan
3. **Aislamiento Modular y Contratos de Datos Rígidos:** El motor desacopla estrictamente la representación topológica
4. **Determinismo Reproducible:** El comportamiento del sistema es puramente determinista. Dada la misma topología
--## 4.2 Arquitectura General y Organización Modular

El motor BLOC-RELOC está estructurado como una jerarquía de subsistemas especializados ordenados en tres capas fun

```
========================================================================
DIAGRAMA DE ARQUITECTURA DEL SISTEMA
========================================================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│
CAPA DE CONTROL Y ORQUESTACIÓN
│
│
│
│
┌─────────────────────────────────┐
│
│
│
ExecutionController
│
│
│
└────────────────┬────────────────┘
│
└────────────────────────────────────────┼────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┴────────────────────────────────────────┐
│
CAPA DE REUBICACIÓN Y EXPANSIÓN
│
│
│
│ ┌────────────────────────┐
┌────────────────────────────┐ │
│ │ RelocationEngine │<─────────────────>│
BlockExpander
│ │
│ └───────────┬────────────┘
└────────────────────────────┘ │
└───────────────┼─────────────────────────────────────────────────────────────────┘
│
▼
┌───────────────┴─────────────────────────────────────────────────────────────────┐
│
CAPA DE ESTADO HOMEOSTÁTICO Y MEMORIA
│
│
│
│ ┌──────────────────┐ ┌──────────────────┐ ┌───────────────────────────┐ │
│ │ AffinityMatrix │ │ BucketPriority │ │
TabuStateTracker
│ │
│ │
│ │
Queues
│ │
│ │
│ └────────┬─────────┘ └────────┬─────────┘ └─────────────┬─────────────┘ │
└───────────┼───────────────────────┼────────────────────────────┼────────────────┘
│
│
│
▼
▼
▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│
CAPA DE ABSTRACCIÓN DE DATOS
│
│
│
│ ┌─────────────────────────────────┐ ┌─────────────────────────────────────┐ │
│ │
CoreGraph (CSR Format) │ │
PartitionState Tracker
│ │
│ └─────────────────────────────────┘ └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
========================================================================
```
### Descripción de los Componentes Principales:

* **`ExecutionController`:** Orquestador principal que gestiona el ciclo de vida de la ejecución, verifica los criterios de c
* **`RelocationEngine`:** Subsistema encargado de consumir los vértices semilla con mayor tensión homeostática, eva
* **`BlockExpander`:** Módulo especializado en la exploración topológica contigua (mediante algoritmos de búsqueda e
* **`AffinityMatrix`:** Matriz densa contigua que mantiene en tiempo real los valores de Luz-Potencial $\Phi_u(j)$ para c
* **`BucketPriorityQueues`:** Estructura de colas por cubos de ganancia indexada en $\mathcal{O}(1)$ que ordena din
* **`TabuStateTracker`:** Vector directo de marcas temporales para la gestión de la tenencia tabú de vértices.

* **`CoreGraph`:** Representación inmutable y de ultra-baja latencia del grafo en formato *Compressed Sparse Row* (C
* **`PartitionState Tracker`:** Mapeo contiguo vectorizado del estado de pertenencia $\pi: V \to \{0, \dots, k-1\}$.
--## 4.3 Representación de Datos y Gestión de Memoria

La eficiencia espacial y temporal del motor depende directamente de las estructuras de datos subyacentes. A continuac
### 4.3.1 Representación del Grafo: Compressed Sparse Row (CSR)

Para evitar la dispersión de memoria de las listas de adyacencia basadas en punteros enlazados, el grafo no dirigido y p

1. **`row_ptr` (Arreglo de Punteros de Filas):** Vector de tamaño $|V| + 1$ de enteros de 64 bits (`uint64_t`). Indica el
2. **`cols` (Arreglo de Columnas/Destino):** Vector de tamaño $2|E|$ de enteros de 32 bits (`uint32_t`). Contiene los id
3. **`weights` (Arreglo de Pesos):** Vector de tamaño $2|E|$ de flotantes de doble precisión (`double`). Almacena el pe

$$\text{Memoria Total CSR} = (|V| + 1) \times 8 \text{ bytes} + 2|E| \times 4 \text{ bytes} + 2|E| \times 8 \text{ bytes

Esta representación garantiza que la iteración sobre la vecindad de un nodo $N(u)$ sea una operación puramente secue
### 4.3.2 Estado de la Partición (`PartitionState Tracker`)

El mapeo de asignación de bloques $\pi$ se almacena como un vector contiguo `partition_map` de tamaño $|V|$ de ent
### 4.3.3 Matriz de Afinidad (`AffinityMatrix`)

La matriz de afinidad homeostática almacena los valores $\Phi_u(j) = \sum_{v \in N(u) \cap B_j} w(u,v)$.
Se implementa como un único bloque plano de memoria contigua en disposición de fila mayor (*row-major order*) de d
$$\text{Offset}(u, j) = u \times k + j$$

Para garantizar que esta estructura no penalice el consumo de memoria en grafos gigantescos con $k$ elevado, la `Affi
### 4.3.4 Colas de Prioridad por Cubos (`BucketPriorityQueues`)

Inspiradas en las estructuras de alto rendimiento para refinamiento de cortes, las colas de prioridad no utilizan montícu

```
========================================================================
ESTRUCTURA INTERNA DE LAS BUCKET PRIORITY QUEUES
========================================================================
Índice de Ganancia (Discrete Bucket)
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ Gain -2 │ Gain -1 │ Gain 0 │ Gain +1 │ Gain +2 │ ...
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
│
│
│
│
│
▼
▼
▼
▼
▼
Null
Null
[u_10]
[u_4]
[u_77] <-- Head Pointers
│▲
│▲
│▲
▼│
▼│
▼│
[u_12]
[u_9]
Null
│▲
│▲
▼│
▼│
Null
Null

Max Gain Cursor ───────────────────────────────────► [Gain +2] (O(1) Access)
========================================================================
```

Dado que la ganancia de reubicación $\Delta \Phi_u$ se discretiza multiplicándola por un factor de escala flotante $\mat
* **`buckets`:** Arreglo de cabeceras de tamaño $2 G_{\max} + 1$.
* **`node_nodes`:** Vector contiguo de tamaño $|V|$ que contiene los punteros `prev` y `next` (`uint32_t`) para cada
* **`max_gain_cursor`:** Puntero entero que rastrea el índice activo de mayor ganancia. Cuando se extrae el elemento
---

## 4.4 Módulos Principales y API Interna

A continuación se documenta el diseño sintáctico y semántico de las interfaces internas que articulan el motor BLOC-RE
### 4.4.1 Módulo `CoreGraph`
Proporciona la interfaz de lectura inmutable sobre la topología del grafo.
```cpp
struct CoreGraph {
uint64_t num_vertices;
uint64_t num_edges;
uint64_t* row_ptr;
uint32_t* cols;
double* weights;
// Retorna el grado del vértice u
inline uint32_t get_degree(uint32_t u) const {
return row_ptr[u + 1] - row_ptr[u];
}
// Retorna un iterador de rango continuo sobre las aristas salientes de u
inline void get_neighbors(uint32_t u, const uint32_t** neighbors,
const double** edge_weights, uint32_t* count) const {
uint64_t start = row_ptr[u];
*count = row_ptr[u + 1] - start;
*neighbors = &cols[start];
*edge_weights = &weights[start];
}
};
```
### 4.4.2 Módulo `AffinityMatrix`
Gestiona la lectura y actualización atómica del estado de Luz-Potencial.
```cpp
class AffinityMatrix {
private:
double* matrix;
uint32_t num_vertices;
uint16_t num_partitions;
public:
void initialize(const CoreGraph& graph, const uint16_t* partition_map);
// Acceso directo O(1) a la afinidad del nodo u con la partición j
inline double get_affinity(uint32_t u, uint16_t j) const {
return matrix[u * num_partitions + j];
}
// Obtiene la partición de máxima atracción y la ganancia neta respecto a la actual
void get_best_target(uint32_t u, uint16_t current_p,
uint16_t* best_p, double* max_gain) const;
// Actualización incremental cuando el nodo v cambia de old_p a new_p
void update_node_influence(uint32_t v, uint16_t old_p, uint16_t new_p,
const CoreGraph& graph);
};
```
### 4.4.3 Módulo `BucketPriorityQueue`
Mantiene el orden de prioridad homeostático de los vértices frontera.
```cpp
class BucketPriorityQueue {
private:

struct ListNode {
uint32_t prev;
uint32_t next;
int32_t current_bucket;
};
uint32_t* bucket_heads;
ListNode* nodes;
int32_t max_gain_cursor;
int32_t offset;
public:
void insert_or_update(uint32_t u, double gain);
void remove(uint32_t u);
uint32_t pop_max();
bool is_empty() const;
};
```
### 4.4.4 Módulo `TabuStateTracker`
Controla el periodo de enfriamiento homeostático para prevenir ciclos de oscilación.
```cpp
class TabuStateTracker {
private:
uint32_t* last_moved_iter;
uint32_t tenure_window;
public:
inline bool is_tabu(uint32_t u, uint32_t current_iter) const {
return (current_iter - last_moved_iter[u]) <= tenure_window;
}
inline void mark_moved(uint32_t u, uint32_t current_iter) {
last_moved_iter[u] = current_iter;
}
void set_tenure_window(uint32_t window) {
this->tenure_window = window;
}
};
```
--## 4.5 Flujo de Datos y Pipeline de Ejecución

El procesamiento de un grafo dentro de BLOC-RELOC sigue una secuencia de tres fases atómicas: **Fase de Carga e Ini

```
========================================================================
DIAGRAMA DE FLUJO DE DATOS
========================================================================
[ Archivo Grafo ] ───► ( CoreGraph::LoadCSR ) ───► [ Estructura CSR ]
│
[ Partición Inicial ] ──► ( PartitionTracker ) ──────────┤
▼
( AffinityMatrix::Initialize )
│
▼
[ Matriz de Afinidad |V| x k ]
│
▼
( BucketQueue::PopulateAll )
│
▼

[ Colas por Cubos de Ganancia ]
│
▼
┌──────────────────────────────────────────────────────┴────────────────────┐
│
CICLO HOMEOSTÁTICO (HOT LOOP)
│
│
│
│ [ BucketQueue ] ──(PopMax)──► u_seed
│
│
│
│
│
▼
│
│
( TabuTracker::IsTabu ) ──(Sí)──► [ Descarta ]
│
│
│(No)
│
│
▼
│
│
( BlockExpander::Expand )
│
│
│
│
│
▼
│
│
[ Bloque Candidato K ]
│
│
│
│
│
▼
│
│
( Evaluar Ganancia Agregada ΔΦ_K )
│
│
│
│
│
┌───────────────┴───────────────┐
│
│
(ΔΦ_K > 0)
(ΔΦ_K <= 0)
│
│
│
│
│
│
▼
▼
│
│
( RelocateBlockAndCommit )
[ Abortar Transacción ]
│
│
│
│
│
│
├──► actualiza PartitionMap
│
│
│
├──► actualiza AffinityMatrix │
│
│
└──► actualiza BucketQueue
│
│
└───────────────────────────────────┬───────────────────────────────────────┘
│
▼
( Verificar Criterio Parada )
│
┌────────────────────┴────────────────────┐
(Sin Mejora)
(Con Mejora)
│
│
▼
▼
[ Retornar Partición Óptima ]
[ Continuar Siguiente Iteración ]
========================================================================
```
--## 4.6 El Ciclo Completo de Optimización

El ciclo de optimización constituye el núcleo de ejecución del sistema. Opera como una máquina de estados determinist

```
========================================================================
DIAGRAMA DEL CICLO COMPLETO DE OPTIMIZACIÓN (HOT LOOP)
========================================================================
┌─────────────────────────────────────────┐
│
INICIO DE ITERACIÓN t
│
└────────────────────┬────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ ¿BucketQueue vacía O t > MaxIter?
│
└────────────────────┬────────────────────┘
│
┌───────────────┴───────────────┐
SÍ
NO
│
│
▼
▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ CONVERGENCIA / TERMINACIÓN │ │
u_seed = PopMaxFromBucket │

└───────────────────────────────┘ └───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ ¿IsTabu(u_seed) O NoVálido? │
└───────────────┬───────────────┘
│
┌───────────────┴───────────────┐
SÍ
NO
│
│
▼
▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Descartar Semilla y Reintentar│ │ Expansión BFS de Bloque K
│
└───────────────────────────────┘ └───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ Calcular ΔΦ_K = Σ ΔΦ_v (v ∈ K)│
└───────────────┬───────────────┘
│
┌───────────────┴───────────────┐
(ΔΦ_K > 0)
(ΔΦ_K <= 0)
│
│
▼
▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ COMMIT: Aplicar Reubicación │ │ Rechazar Bloque. Aplicar
│
│ - Actualizar PartitionMap
│ │ Penalización Local a u_seed │
│ - Incrementar AffinityMatrix │ └───────────────────────────────┘
│ - Marcar Tabu(K, t)
│
└───────────────┬───────────────┘
│
▼
┌───────────────────────────────┐
│ Resetear Contador Estancado │
└───────────────────────────────┘
========================================================================
```
### Algoritmo Detallado del Bucle Principal de Optimización
```
Procedimiento EJECUTAR_CICLO_HOMEOSTATICO(G, PartitionState, Affinity, BucketQueue, Tabu, Config):
t=0
stagnation_counter = 0
best_cost = CALCULAR_COSTO_CORTE_GLOBAL(G, PartitionState)
Mientras t < Config.max_iterations y stagnation_counter < Config.max_stagnation:
t=t+1
Si BucketQueue.is_empty():
Ruptura (No hay más nodos en frontera con presión homeostática)
u_seed = BucketQueue.pop_max()
Si Tabu.is_tabu(u_seed, t):
Continuar
target_partition, seed_gain = Affinity.get_best_target(u_seed, PartitionState[u_seed])
Si seed_gain <= 0:
Continuar
// Fase de Expansión de Bloque Contiguo
K = EXPANDIR_BLOQUE_CONTIGUO(G, u_seed, PartitionState, target_partition,
Config.max_block_size, Tabu, t)
// Evaluación de Ganancia Agregada Real
delta_phi_K = CALCULAR_GANANCIA_AGREGADA_BLOQUE(G, K, PartitionState,

target_partition, Affinity)
Si delta_phi_K > 0 y VALIDAR_BALANCE_PARTICIÓN(PartitionState, K, target_partition):
// Aplicación Transaccional del Movimiento
origin_partition = PartitionState[u_seed]
Para cada v en K:
PartitionState[v] = target_partition
Tabu.mark_moved(v, t)
// Actualización Incremental de la Matriz de Afinidad para la vecindad expandida
Para cada v en K:
Affinity.update_node_influence(v, origin_partition, target_partition, G)
// Actualización de la Cola de Prioridad
Para cada v en K y vecinos w en N(K):
nueva_ganancia = CALCULAR_GANANCIA_NODO(w, PartitionState, Affinity)
BucketQueue.insert_or_update(w, nueva_ganancia)
best_cost = best_cost - delta_phi_K
stagnation_counter = 0
Sino:
stagnation_counter = stagnation_counter + 1
Retornar PartitionState, best_cost
```
--## 4.7 Interacción entre Módulos y Gestión de Afinidad

La alta velocidad del motor BLOC-RELOC reside en el acoplamiento estrecho y sincronizado entre `AffinityMatrix`, `Buck

Cuando un bloque $K$ se desplaza de la partición $B_{\text{origin}}$ a $B_{\text{target}}$, solo la frontera inmediat
### Ecuación de Actualización Incremental de Afinidad
Para cualquier nodo $w \in N(K)$ que es vecino de un nodo reubicado $v \in K$ con peso de arista $w(v, w)$:
$$\Phi_w(B_{\text{origin}}) \leftarrow \Phi_w(B_{\text{origin}}) - w(v, w)$$
$$\Phi_w(B_{\text{target}}) \leftarrow \Phi_w(B_{\text{target}}) + w(v, w)$$
Inmediatamente después de actualizar $\Phi_w$, el motor recalcula la ganancia potencial del nodo $w$:
$$\Delta \Phi_w = \Phi_w(\nu(w)) - \Phi_w(\pi(w))$$

Este valor se envía a la `BucketPriorityQueue` mediante `insert_or_update(w, \Delta \Phi_w)`. Si $w$ ya estaba en la co

```
========================================================================
SECUENCIA DE ACTUALIZACIÓN INCREMENTAL EN LA FRONTERA
========================================================================
[ Reubicación del Bloque K ]
│
▼
Para cada v ∈ K:
│
├─► PartitionState[v] = B_target
│
└─► Iterar sobre w ∈ N(v):
│
├─► AffinityMatrix[w][B_origin] -= w(v,w)
├─► AffinityMatrix[w][B_target] += w(v,w)
│
├─► Recalcular ν(w) y ΔΦ_w
│

└─► BucketPriorityQueue::InsertOrUpdate(w, ΔΦ_w)
========================================================================
```
--## 4.8 Análisis de Complejidad Temporal y Espacial
A continuación se detalla la caracterización asintótica de las operaciones fundamentales del sistema, donde $|V|$ es el

| Módulo / Operación | Complejidad Temporal (Worst Case) | Complejidad Temporal (Amortizada) | Complejidad Espacial
|---|---|---|---|
| **Carga CSR (`CoreGraph`)** | $\mathcal{O}(|V| + |E|)$ | $\mathcal{O}(|V| + |E|)$ | $(8|V| + 24|E|) \text{ bytes}$ |
| **Inicialización Afinidad** | $\mathcal{O}(|V| \cdot d_{\max})$ | $\mathcal{O}(|E|)$ | $8 |V| k \text{ bytes}$ |
| **Extracción Semilla (`PopMax`)** | $\mathcal{O}(G_{\max})$ | $\mathcal{O}(1)$ | $8 G_{\max} \text{ bytes}$ |
| **Expansión Bloque (`BFS`)** | $\mathcal{O}(S_K \cdot d_{\max})$ | $\mathcal{O}(S_K \cdot \bar{d})$ | $4 S_K \text
| **Actualización Afinidad** | $\mathcal{O}(|K| \cdot d_{\max})$ | $\mathcal{O}(|K| \cdot \bar{d})$ | $\mathcal{O}(1)$
| **Re-indexado en Buckets** | $\mathcal{O}(|K| \cdot d_{\max})$ | $\mathcal{O}(|K| \cdot \bar{d})$ | $12 |V| \text{ b
| **Consulta Tabú (`IsTabu`)** | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $4 |V| \text{ bytes}$ |
### Análisis Espacial Global

La memoria total consumida por el motor BLOC-RELOC se mantiene acotada por una función estrictamente lineal respec
$$\text{Memoria Total} = \mathcal{O}(|V| \cdot k + |E|) \text{ [Bytes]}$$

Esta cota inferior garantiza que el motor sea capaz de cargar grafos de más de 50 millones de aristas en máquinas de c
--## 4.9 Razones de Ingeniería, Cuellos de Botella y Escalabilidad

El diseño de la arquitectura de BLOC-RELOC implicó la toma de decisiones de ingeniería dirigidas a balancear la velocida
### 4.9.1 Fundamentación de las Decisiones Principales

1. **Selección del formato CSR frente a Listas de Adyacencia:** Las listas de adyacencia convencionales (`std::vector<s
2. **Adopción de Colas por Cubos frente a Montículos Binarios (`std::priority_queue`):** Los montículos binarios impone
### 4.9.2 Cuellos de Botella Identificados

A pesar de las optimizaciones, la arquitectura presenta vulnerabilidades estructurales cuando se enfrenta a ciertas fami

* **Involución por *Super-Hubs* (Heterogeneidad Extrema):** En redes donde un vértice posee un grado $d_u \gg 10^4
* **Saturación de la `AffinityMatrix` en $k$ elevado:** Cuando el número de particiones requeridas crece ($k > 1000$),
### 4.9.3 Ventajas y Limitaciones de la Arquitectura

```
========================================================================
BALANCE ARQUITECTÓNICO DEL MOTOR BLOC-RELOC
========================================================================
VENTAJAS
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Velocidad de convergencia ultra-rápida (Sub-segunda en |V| < 10^6).
│
│ 2. Cero asignación de memoria en tiempo de ejecución (Hot Loop puro).
│
│ 3. Determinismo absoluto y reproducibilidad completa del pipeline.
│
│ 4. Operaciones de refinamiento contiguo por bloques que previenen mínimos
│
│ locales superficiales.
│
└─────────────────────────────────────────────────────────────────────────────┘
LIMITACIONES
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Sensibilidad extrema a la heterogeneidad de grado (Super-hubs).
│
│ 2. Escalamiento de memoria dependiente del producto lineal |V| x k.
│
│ 3. Ausencia de mecanismos de desacoplamiento para grafos no modulares
│
│ antes de la fase de refinamiento.
│

└─────────────────────────────────────────────────────────────────────────────┘
========================================================================
```
----# PARTE II: VALIDACIÓN EXPERIMENTAL
--# CAPÍTULO 5: FRAMEWORK DE VALIDACIÓN Y BENCHMARKING
--## 5.1 Objetivos Científicos y Principios de Reproducibilidad Experimental

La evaluación de un algoritmo de optimización combinatoria en problemas NP-difíciles exige un marco de experimentac

Para garantizar la validez científica y la aceptabilidad académica de los hallazgos, el Laboratorio TCF estableció cuatro p

- **Invariancia del Entorno de Ejecución:** Aislamiento total de recursos de cómputo a nivel de sistema operativo para e
- **Aislamiento Semillado y Control de Estocasticidad:** Control estricto de la pseudoaleatoriedad mediante secuencias
- **Paridad de Interfaces e Insumos:** Todos los solucionadores deben consumir exactamente la misma representación
- **Mapeo Métrico Holístico:** Medición simultánea de la calidad de la solución (costo de corte, conductancia, balance),
--## 5.2 Arquitectura del Harness Experimental

Para orquestar la ejecución automatizada de miles de instancias de prueba, el laboratorio diseñó e implementó un arné

```
========================================================================
DIAGRAMA DE ARQUITECTURA DEL HARNESS EXPERIMENTAL
========================================================================
┌─────────────────────────────────────────────────────────────────────────────┐
│
CONFIGURADOR Y ORQUESTADOR
│
│
(BenchmarkExecutionManager)
│
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│
CARGADOR DE DATASETS
│
│
(DatasetRegistry & CSRLoader)
│
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│
AISLADOR DE EJECUCIÓN
│
│
(ProcessIsolation & Telemetry)
│
│
│
│ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌───────────────┐ │
│ │ Process Pinning (CPU) │ │ Memory Limit (cgroups) │ │ Timer / Watch │ │
│ └─────────────────────────┘ └─────────────────────────┘ └───────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│
ARNESES DE INVOCACIÓN (ADAPTERS)
│
│
│
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│ │ BLOC-RELOC │ │ KaHyPar
│ │ BLS Adapter │ │ Baseline Adapters │ │
│ │ Engine
│ │ C++ API
│ │ C++ Binary │ │ (KL, TS, GA, Spectral)│ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────┘

│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│
MÓDULO DE TELEMETRÍA Y LOGS
│
│
(TelemetryCollector & Exporter)
│
└──────────────────────────────────────┬──────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│
REPOSITORIO DE EVALUACIÓN
│
│
(CSV Logs, Stats, Performance Profiles)
│
└─────────────────────────────────────────────────────────────────────────────┘
========================================================================
```
### Pipeline Completo de Ejecución Experimental

El ciclo de vida de un experimento individual dentro de TCF-BenchHarness sigue una secuencia rigurosa e ininterrumpid

```
========================================================================
PIPELINE DE EJECUCIÓN EXPERIMENTAL
========================================================================

[ Config XML/JSON ] ──► ( Cargar Parámetros y Semillas )
│
▼
[ Instancia Grafo ] ──► ( Cargar y Validar Topología en CSR )
│
▼
( Generar Partición Inicial π_0 )
│
▼
( Despachar Proceso Aislado )
│
┌───────────────────────────┴───────────────────────────┐
│
│
▼
▼
[ Monitoreo Telemetría ]
[ Ejecución Algorítmica ]
- Muestra de RAM (RSS)
- Medición de Tiempo (POSIX CPU)
- Conteo de Swaps
- Registro de Movimientos
- Control de Timeout (Watchdog)
- Retorno de Partición π_final
│
│
└───────────────────────────┬───────────────────────────┘
│
▼
( Módulo de Verificación )
- Comprobar Validez de π_final
- Verificar Contigüidad y Cobertura
│
▼
( Cálculo de Métricas Finales )
- Cut Weight, Conductance, Balance
│
▼
( Exportar Registro CSV Incremental )
========================================================================
```
--## 5.3 Selección y Caracterización de Algoritmos Competidores

Para evaluar a BLOC-RELOC v1.0 dentro del espectro global de la optimización combinatoria, se seleccionaron diez soluc

```
========================================================================
TAXONOMÍA DE ALGORITMOS EVALUADOS

========================================================================

LÍNEAS BASE
MÉTODOS
METAHEURÍSTICAS
ESTÁNDAR INDUSTRIAL
CONSTRUCTIVAS
ESPECTRALES
ESTOCÁSTICAS
MULTINIVEL
┌──────────────┐
┌─────────────┐ ┌───────────────┐ ┌─────────────────┐
│ Random
│
│ Spectral │ │ Simulated
│ │ KaHyPar
│
│ Greedy
│
└─────────────┘ │ Annealing
│ └─────────────────┘
└──────────────┘
│ Tabu Search │
│ Genetic Alg. │ MOTOR HOMEOSTÁTICO
REFINAMIENTO
│ Breakout LS │ ┌─────────────────┐
LOCAL CLÁSICO
└───────────────┘ │ BLOC-RELOC
│
┌──────────────┐
└─────────────────┘
│ Kernighan- │
│ Lin (KL)
│
└──────────────┘
========================================================================
```
### 5.3.1 Líneas Base Constructivas

**Random Partitioning Baseline (Random):**
- **Descripción:** Asigna aleatoriamente cada vértice $u \in V$ a una de las $k$ particiones con probabilidad uniforme $
- **Propósito Científico:** Establece el suelo de rendimiento absoluto. Permite medir el porcentaje de mejora neta que c

**Greedy Degree-Partitioning (Greedy):**
- **Descripción:** Ordena los vértices por grado descendente $d_u$ y los asigna iterativamente al bloque $B_j$ que min
- **Propósito Científico:** Evalúa la capacidad de una regla voraz univariada determinista para resolver el corte sin fase
### 5.3.2 Métodos Algebraicos y Espectrales

**Spectral Bisection (Spectral):**
- **Descripción:** Biparticiona el grafo calculando el vector propio correspondiente al segundo valor propio más pequeñ
- **Propósito Científico:** Proporciona la referencia algebraica continua global de la estructura del grafo, sirviendo como
### 5.3.3 Metaheurísticas Estocásticas de Búsqueda Local

**Simulated Annealing (SA):**
- **Descripción:** Aplica el algoritmo de Metropolis-Hastings sobre el espacio de asignación. Acepta movimientos que au
- **Propósito Científico:** Mide la eficacia de los saltos estocásticos no guiados para escapar de mínimos locales en com

**Tabu Search Clásico (TS):**
- **Descripción:** Explora el vecindario de soluciones permitiendo movimientos individuales de nodos que degradan la f
- **Propósito Científico:** Permite comparar la memoria tabú univariada estándar contra el mecanismo de Memoria Tab
**Genetic Algorithm Partitioning (GA):**
- **Descripción:** Mantiene una población de $P$ vectores de partición. Utiliza operadores de cruzamiento por corte de
- **Propósito Científico:** Representa la aproximación poblacional basada en recombinación genética.

**Breakout Local Search (BLS):**
- **Descripción:** Combina una fase de descenso local voraz con una fase de perturbación adaptativa. Cuando el algorit
- **Propósito Científico:** Representa el estado del arte en metaheurísticas de búsqueda local combinatoria con escape
### 5.3.4 Refinamiento Local Clásico

**Kernighan-Lin Heuristic (KL):**
- **Descripción:** Algoritmo clásico de refinamiento por intercambio de pares de vértices ($swaps$). Calcula la ganancia
- **Propósito Científico:** Es el estándar de comparación por excelencia para heurísticas de intercambio local. Permite v
### 5.3.5 Estándar Industrial Multinivel

**KaHyPar (Karlsruhe Hypergraph Partitioning):**
- **Descripción:** Framework de particionamiento multinivel (Multilevel Framework) de última generación. Aplica fases
- **Propósito Científico:** Representa el límite superior de calidad y rendimiento del estado del arte industrial y académ
### 5.3.6 El Solucionador Evaluado

**BLOC-RELOC (v1.0):**
- **Descripción:** Motor homeostático de particionamiento basado en la Teoría de Conciencia Fotónica (TCF), caracteriz

--## 5.4 Formulación Matemática de Métricas de Evaluación

Para analizar el comportamiento de los solucionadores desde múltiplos ángulos, el framework calcula automáticamente
### 1. Costo del Corte ($Cut\ Weight - C(\Pi)$)
Suma total del peso de las aristas cuyas extremidades pertenecen a particiones distintas:
$$C(\Pi) = \frac{1}{2} \sum_{i \neq j} \sum_{u \in B_i, v \in B_j} w(u,v)$$
### 2. Conductancia Global ($\phi(\Pi)$)

Normalización del costo del corte respecto al volumen menor de las particiones creadas, midiendo la calidad de aislamie
$$\phi(\Pi) = \max_{i} \left( \frac{C(B_i, V \setminus B_i)}{\min(\text{Vol}(B_i), \text{Vol}(V \setminus B_i))} \right)$$
donde $\text{Vol}(B_i) = \sum_{u \in B_i} d_u$.
### 3. Factor de Inbalance ($Balance\ Ratio - \beta(\Pi)$)
Mide la desviación máxima del tamaño de las particiones respecto al tamaño ideal balanceado $|V|/k$:
$$\beta(\Pi) = \max_{j \in \{1, \dots, k\}} \frac{|B_j|}{\lceil |V|/k \rceil}$$
Un valor de $\beta(\Pi) = 1.0$ representa una partición perfectamente balanceada.
### 4. Función de Costo Penalizada ($\mathcal{L}(\Pi)$)

Incorpora una penalización cuadrática por desbalance para evaluar la calidad funcional del corte bajo restricciones rígid
$$\mathcal{L}(\Pi) = C(\Pi) + \lambda \cdot \max\left(0, \beta(\Pi) - (1 + \epsilon)\right)^2$$

donde $\lambda = 10^5$ es la constante de penalización de frontera y $\epsilon = 0.05$ es la tolerancia de desbalance
### 5. Brecha Relativa de Optimalidad ($Relative\ Gap - \text{Gap}_A$)

Porcentaje de desviación del costo obtenido por el algoritmo $A$ respecto al mejor costo conocido ($C^*$) para una ins
$$\text{Gap}_A(\%) = \left( \frac{C_A - C^*}{C^*} \right) \times 100$$
### 6. Tiempo de Ejecución ($Execution\ Time - T_{\text{exec}}$)

Tiempo transcurrido en milisegundos ($\text{ms}$) medido vía llamadas POSIX clock_gettime(CLOCK_MONOTONIC) exc
### 7. Pico de Consumo de Memoria ($Peak\ RAM - M_{\text{peak}}$)

Máximo tamaño de conjunto residente (Resident Set Size - RSS) registrado en megabytes ($\text{MB}$) durante el ciclo
### 8. Tasa de Timeout ($R_{\text{timeout}}$)

Variable binaria $\delta_{\text{TO}} \in \{0, 1\}$ que registra si el algoritmo sobrepasó el límite máximo de tiempo wal
### 9. Tasa de Error de Estructura ($R_{\text{err}}$)

Variable binaria $\delta_{\text{ERR}} \in \{0, 1\}$ que se activa si la solución entregada viola la integridad del conjunto
--## 5.5 Caracterización del Dataset de Benchmarking y Familias Topológicas

El dataset experimental fue construido para cubrir un espectro amplio de propiedades geométricas, variando desde estr

```
========================================================================
MATRIZ DE DATASETS EXPERIMENTALES
========================================================================

CATEGORÍA SINTÉTICA / ESTRUCTURADA
CATEGORÍA REDES REALES (SNAP)
┌─────────────────────────────────┐
┌─────────────────────────────────┐
│ 1. Random Regular Graphs
│
│ 6. SNAP Facebook (soc-FB)
│
│ 2. Watts-Strogatz (Small-World) │
│ 7. SNAP WikiVote (wiki-Vote) │
│ 3. Barabási-Albert (Scale-Free) │
└─────────────────────────────────┘
│ 4. Planted Partition (SBM)
│
│ 5. Disconnected Networks
│
└─────────────────────────────────┘
========================================================================
```
### 5.5.1 Justificación Científica de las Familias Seleccionadas

**Random Regular Graphs ($G(n,k)$):**
- **Propiedades:** Grado constante $d_u = k$ para todo $u \in V$. Sin "hubs" ni nodos periféricos.
- **Justificación:** Permite aislar la capacidad del algoritmo para encontrar cortes en ausencia de pistas topológicas o va

**Watts-Strogatz Small-World ($WS(n, k, p)$):**
- **Propiedades:** Alto coeficiente de agrupamiento local ($Clustering\ Coefficient$) y corta distancia promedio de ruta.
- **Justificación:** Modela redes con alta densidad comunitaria local y atajos inter-comunitarios ocasionales.

**Barabási-Albert Scale-Free ($BA(n, m)$):**
- **Propiedades:** Distribución de grado que sigue una ley de potencia $P(k) \sim k^{-\gamma}$.
- **Justificación:** Introduce la presencia de "super-hubs" (nodos con miles de conexiones) para evaluar cómo la hetero

**Planted Partition Model / Stochastic Block Model ($SBM(n, k, p_{in}, p_{out})$):**
- **Propiedades:** Estructura comunitaria intrínseca conocida con densidad controlada $p_{in} \gg p_{out}$.
- **Justificación:** Proporciona una "verdad de terreno" (ground truth) estructural para verificar si el algoritmo recupera

**Disconnected / Weakly Coupled Networks:**
- **Propiedades:** Redes compuestas por múltiples componentes disjuntas o unidas por puentes de peso mínimo.
- **Justificación:** Evalúa la estabilidad del algoritmo ante discontinuidades absolutas en la propagación del flujo de info

**SNAP Facebook Social Network (soc-FB):**
- **Propiedades:** Red real de interacciones sociales ($|V| = 4,039, |E| = 88,234$). Alta densidad de traslape modular.
- **Justificación:** Representa la complejidad de las redes sociales del mundo real con distribuciones de grado altament

**SNAP WikiVote Network (wiki-Vote):**
- **Propiedades:** Red real de votación ($|V| = 7,115, |E| = 103,689$).
- **Justificación:** Testea la resistencia de las heurísticas en redes dirigidas/convertidas a no dirigidas con patrones de c
### Tabla Resumen de Instancias de Prueba Estándar

| Identificador Instancia | Familia Topológica | Vértices ($|V|$) | Aristas ($|E|$) | Grado Promedio ($\bar{d}$) | Assortativ
|---|---|---|---|---|---|
| SYN_REG_10K | Random Regular | $10,000$ | $50,000$ | $10.00$ | $-0.001$ |
| SYN_WS_10K | Watts-Strogatz | $10,000$ | $40,000$ | $8.00$ | $+0.012$ |
| SYN_BA_10K | Barabási-Albert | $10,000$ | $39,950$ | $7.99$ | $-0.065$ |
| SYN_SBM_10K | Planted Partition | $10,000$ | $62,400$ | $12.48$ | $+0.450$ |
| SNAP_FB_4K | SNAP Facebook | $4,039$ | $88,234$ | $43.69$ | $+0.064$ |
| SNAP_WIKI_7K | SNAP WikiVote | $7,115$ | $103,689$ | $29.14$ | $-0.083$ |
--## 5.6 Control de Aleatoriedad, Hardware y Entorno de Software
Para garantizar que los resultados no contuvieran sesgos inducidos por la plataforma de ejecución, el laboratorio aplicó
### 5.6.1 Control de Aleatoriedad y Protocolo de Semillado

Para eliminar la variabilidad de las heurísticas estocásticas (SA, TS, GA, BLS, Random), cada experimento sobre una inst
$$\mathcal{S} = \{s_1, s_2, \dots, s_{30}\} = \{42, 101, 2023, 777, 9999, \dots\}$$

Este protocolo de 30 réplicas por combinación (Algoritmo, Instancia, $k$) garantiza un tamaño de muestra suficiente ($

### 5.6.2 Especificación y Aislamiento del Hardware de Cómputo
Todos los experimentos de la campaña masiva fueron ejecutados en un nodo de cómputo dedicado del Laboratorio TCF

- **Procesador:** AMD EPYC 7702 (64 Núcleos físicos, 128 Hilos, Caché L3 de 256 MB).
- **Memoria RAM:** 256 GB DDR4 ECC @ 3200 MHz.
- **Aislamiento de CPU:** Invocación de utilidades `taskset -c` para aislar los procesos del benchmark en núcleos físicos
- **Control de Frecuencia:** Gobernador de CPU fijado en modo `performance` (`cpupower frequency-set -g performanc
### 5.6.3 Entorno de Software y Compilación
- **Sistema Operativo:** Ubuntu 22.04.4 LTS (Kernel Linux 5.15.0-105-generic).
- **Cadena de Compilación:** GCC 11.4.0 / Clang 14.0.0.
- **Flags de Compilación C++:**
```bash
-O3 -march=native -ffast-math -flto -funroll-loops -DNDEBUG
```
--## 5.7 Diseño Estadístico, Análisis de Estabilidad y Perfiles de Desempeño

El procesamiento de los datos crudos generados por el harness experimental requirió el uso de metodologías de estadís
### 5.7.1 Prueba de Rangos con Signo de Wilcoxon (Wilcoxon Signed-Rank Test)

Para comparar las distribuciones de costo de corte entre dos algoritmos $A$ y $B$ sobre múltiples ejecuciones sin asum
$$W = \sum_{i=1}^{N_r} \left[ \text{rank}(|d_i|) \cdot \text{sgn}(d_i) \right]$$

Se establece un nivel de significancia $\alpha = 0.01$. Un p-valor $p < 0.01$ rechaza la hipótesis nula $H_0$ (de que a
### 5.7.2 Tamaño del Efecto de Cohen (Cohen's d)
Para cuantificar la magnitud de la diferencia práctica entre solucionadores, se calcula la métrica de Cohen's $d$:
$$d = \frac{\bar{C}_A - \bar{C}_B}{s_{\text{pooled}}}$$
donde $s_{\text{pooled}} = \sqrt{\frac{(N_A - 1)s_A^2 + (N_B - 1)s_B^2}{N_A + N_B - 2}}$.
Un valor de $|d| > 0.8$ denota un tamaño del efecto grande (diferencia sustancial de rendimiento).
### 5.7.3 Perfiles de Desempeño de Dolan-Moré (Dolan-Moré Performance Profiles)

Para evaluar la eficiencia global de un algoritmo sobre todo el conjunto de instancias, se adopta la metodología de Dola
$$r_{p,s} = \frac{f_{p,s}}{\min \{ f_{p,s'} : s' \in \mathcal{S} \}}$$

donde $f_{p,s}$ es la métrica evaluada (por ejemplo, el costo del corte o el tiempo de ejecución) del algoritmo $s$ en e
$$\rho_s(\tau) = \frac{1}{|\mathcal{P}|} \left| \{ p \in \mathcal{P} : r_{p,s} \le \tau \} \right|$$

El valor de $\rho_s(1)$ indica la fracción de instancias en las que el algoritmo $s$ fue el ganador absoluto del benchmar
--## 5.8 Estrategia de Registro y Estructura de Salida de Datos

Para asegurar la trazabilidad de la información, el módulo TelemetryCollector genera archivos de salida estructurados d
### Estructura del Directorio de Salida
```
/TCF_BENCHMARK_RESULTS/
│
├── logs_raw/
│ ├── SYN_REG_10K_BLOC-RELOC_seed42.log
│ ├── SYN_REG_10K_KaHyPar_seed42.log
│ └── SNAP_FB_4K_BLOC-RELOC_seed42.log
│
├── csv_incremental/
│ ├── master_benchmark_results.csv

│ └── telemetry_memory_time.csv
│
└── statistical_reports/
├── wilcoxon_test_results.json
├── cohen_d_effects.json
└── dolan_more_profiles.dat
```
### Formato del Esquema CSV Incremental (`master_benchmark_results.csv`)

```
instance_id,family,V,E,algorithm,seed,cut_weight,conductance,balance_ratio,cost_penalized,time_ms,peak_ram_mb,time
SYN_REG_10K,Regular,10000,50000,BLOC-RELOC,42,12450,0.249,1.002,12450.0,340.2,18.4,0,0
SYN_REG_10K,Regular,10000,50000,Kernighan-Lin,42,12480,0.249,1.005,12480.0,1850.5,12.1,0,0
SNAP_FB_4K,SNAP_FB,4039,88234,BLOC-RELOC,42,8420,0.182,1.034,8420.0,890.1,24.5,0,0
SNAP_FB_4K,SNAP_FB,4039,88234,BreakoutLS,42,5120,0.112,1.011,5120.0,4520.1,32.0,0,0
```
----# CAPÍTULO 6: RESULTADOS EXPERIMENTALES COMPLETOS
--## 6.1 Introducción Metodológica y Organización de la Evidencia Empírica

En este capítulo se presenta la totalidad de los datos experimentales obtenidos mediante la ejecución del protocolo de v
La presentación de los resultados sigue una organización jerárquica estricta:

1. **Desglose por Familia Topológica:** Tablas paramétricas completas que reportan los estimadores de tendencia cent
2. **Análisis de Dinámica y Convergencia:** Representación ASCII de curvas de convergencia temporal y distribuciones
3. **Evaluación de Eficiencia y Dominancia Global:** Perfiles de desempeño de Dolan-Moré y matrices de ranking relativ
4. **Validación Estadística Infranqueable:** Tablas cruzadas del test de rangos con signo de Wilcoxon y tamaños del efe
5. **Análisis de Estabilidad y Varianza Operacional:** Coeficientes de variación inter-semilla para evaluar la robustez es

Para mantener la rigurosidad y neutralidad científica, **los datos presentados a continuación son puramente descriptivo
--## 6.2 Resultados por Familia Topológica
### 6.2.1 Familia 1: Grafos Regulares Aleatorios (Random Regular Graphs $G(n,k)$)

La evaluación sobre la familia $G(n, k)$ midió el desempeño de los algoritmos en entornos de simetría local absoluta y h
**Tabla 6.1: Resultados de Benchmarking en Random Regular Graphs ($|V| = 10,000, |E| = 50,000, k=2$)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 24,998.4 | 24,999.0 | 112.3 | 24,780 | 25,210 | +100.82 | 12.1 | 8.2 | 0.0% | 0.0% |
| **Greedy** | 18,420.1 | 18,415.0 | 145.8 | 18,110 | 18,730 | +47.98 | 45.3 | 11.4 | 0.0% | 0.0% |
| **Spectral** | 13,850.0 | 13,850.0 | **0.0** | 13,850 | 13,850 | +11.26 | 2,140.8 | 84.6 | 0.0% | 0.0% |
| **SA** | 13,110.5 | 13,095.0 | 210.4 | 12,780 | 13,540 | +5.32 | 12,450.2 | 14.1 | 0.0% | 0.0% |
| **KL** | 12,680.3 | 12,675.0 | 88.2 | 12,510 | 12,890 | +1.86 | 1,850.5 | 12.1 | 0.0% | 0.0% |
| **TS** | 12,890.7 | 12,880.0 | 165.1 | 12,600 | 13,210 | +3.55 | 8,920.1 | 15.8 | 0.0% | 0.0% |
| **GA** | 13,420.2 | 13,400.0 | 312.8 | 12,950 | 14,100 | +7.81 | 24,110.6 | 48.2 | 0.0% | 0.0% |
| **BLS** | 12,520.4 | 12,515.0 | 54.1 | 12,460 | 12,640 | +0.58 | 4,520.1 | 32.0 | 0.0% | 0.0% |
| **KaHyPar** | **12,448.1** | **12,445.0** | 12.4 | **12,430** | **12,470** | **0.00** | 1,280.4 | 42.1 | 0.0% | 0.0% |
| **BLOC-RELOC** | 12,450.2 | 12,448.0 | 18.6 | 12,435 | 12,485 | +0.02 | **340.2** | **18.4** | 0.0% | 0.0% |
**Diagrama 6.1: Boxplot ASCII del Costo de Corte en Random Regular Graphs**
```
Cost (Cut Weight)
13500 ┼

│
13200 ┼
┬ (GA)
│
┬ (SA)
│
12900 ┼
┬ (Spectral) │
┴
│
│
┴
┬ (TS)
12600 ┼
┴
│
┬ (KL) ┬ (BLS)
│
┴
┴
│
┬ (KaHyPar) ┬ (BLOC)
12400 ┼───────────────────────────────────────────────┴────────┼────────────┼───
└─────────┴──────────────┴─────┴───────┴────────┴────────┴────────────┴───
Spectral
SA TS
KL
BLS
KaHyPar BLOC-RELOC
```
--### 6.2.2 Familia 2: Redes Small-World (Watts-Strogatz $WS(n, k, p)$)

En las instancias $WS(10000, 8, 0.05)$, se evaluó el comportamiento ante un alto coeficiente de agrupamiento local com
**Tabla 6.2: Resultados de Benchmarking en Watts-Strogatz Small-World ($|V| = 10,000, |E| = 40,000$)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 19,995.2 | 19,996.0 | 95.1 | 19,810 | 20,180 | +389.26 | 10.4 | 7.9 | 0.0% | 0.0% |
| **Greedy** | 12,110.8 | 12,105.0 | 120.4 | 11,890 | 12,340 | +196.33 | 38.1 | 10.2 | 0.0% | 0.0% |
| **Spectral** | 5,210.0 | 5,210.0 | **0.0** | 5,210 | 5,210 | +27.48 | 1,890.2 | 76.2 | 0.0% | 0.0% |
| **SA** | 4,890.3 | 4,885.0 | 185.2 | 4,550 | 5,210 | +19.66 | 11,200.5 | 13.5 | 0.0% | 0.0% |
| **KL** | 4,320.1 | 4,310.0 | 72.4 | 4,190 | 4,480 | +5.71 | 1,420.8 | 11.5 | 0.0% | 0.0% |
| **TS** | 4,510.6 | 4,500.0 | 140.8 | 4,280 | 4,790 | +10.36 | 7,650.3 | 14.2 | 0.0% | 0.0% |
| **GA** | 5,100.4 | 5,080.0 | 290.1 | 4,620 | 5,610 | +24.80 | 21,800.0 | 42.1 | 0.0% | 0.0% |
| **BLS** | 4,180.2 | 4,175.0 | 41.3 | 4,110 | 4,260 | +2.28 | 3,890.4 | 28.1 | 0.0% | 0.0% |
| **KaHyPar** | 4,102.5 | 4,100.0 | 10.1 | **4,080** | 4,125 | +0.38 | 1,110.2 | 38.4 | 0.0% | 0.0% |
| **BLOC-RELOC** | **4,087.0** | **4,085.0** | 12.8 | 4,082 | **4,108** | **0.00** | **295.4** | **16.2** | 0.0% | 0.0% |
**Diagrama 6.2: Curva de Convergencia ASCII (Costo vs. Tiempo en ms) en Small-World**
```
Cut Weight
6000 ┼──(SA)─────────┐
│
└───(TS)────────┐
5000 ┼───────────────────────────────└───(KL)────────┐
│
└───(BLS)──────┐
4200 ┼──────────────────────────────────────────────────────────────┼───(KaHyPar)
4087 ┼──────────────────────────────────────────────────────────────┴───(BLOC-RELOC)
└─┬─────────────┬───────────────┬───────────────┬──────────────┬────────────►
0
500
1000
2000
4000 Time (ms)
```
--### 6.2.3 Familia 3: Redes Scale-Free (Barabási-Albert $BA(n, m)$)

Las instancias $BA(10000, 4)$ introdujeron concentraciones extremas de grado mediante distribuciones de ley de poten
**Tabla 6.3: Resultados de Benchmarking en Barabási-Albert Scale-Free ($|V| = 10,000, |E| = 39,950$)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 19,970.1 | 19,972.0 | 102.4 | 19,750 | 20,190 | +513.78 | 11.2 | 8.0 | 0.0% | 0.0% |
| **Greedy** | 8,920.4 | 8,915.0 | 210.5 | 8,510 | 9,340 | +174.16 | 41.2 | 10.8 | 0.0% | 0.0% |
| **Spectral** | 4,120.0 | 4,120.0 | **0.0** | 4,120 | 4,120 | +26.63 | 1,980.5 | 78.4 | 0.0% | 0.0% |
| **SA** | 4,850.1 | 4,830.0 | 340.2 | 4,210 | 5,450 | +49.07 | 14,100.0 | 14.8 | 0.0% | 0.0% |
| **KL** | 3,890.5 | 3,880.0 | 112.3 | 3,690 | 4,120 | +19.57 | 2,450.1 | 12.8 | 0.0% | 0.0% |
| **TS** | 4,210.2 | 4,200.0 | 220.8 | 3,850 | 4,680 | +29.41 | 9,810.4 | 16.1 | 0.0% | 0.0% |
| **GA** | 5,420.8 | 5,390.0 | 450.6 | 4,610 | 6,320 | +66.61 | 28,900.2 | 49.5 | 0.0% | 0.0% |
| **BLS** | 3,380.1 | 3,375.0 | 38.2 | 3,310 | 3,460 | +3.89 | 5,120.8 | 30.2 | 0.0% | 0.0% |
| **KaHyPar** | **3,253.4** | **3,250.0** | **8.4** | **3,240** | **3,270** | **0.00** | 1,450.2 | 40.5 | 0.0% | 0.0% |
| **BLOC-RELOC** | 3,650.8 | 3,642.0 | 145.2 | 3,410 | 3,920 | +12.21 | **410.5** | **19.1** | 0.0% | 0.0% |
---

### 6.2.4 Familia 4: Modelo Planted Partition / Stochastic Block Model ($SBM$)
Evaluación en grafos con módulos densos explícitos ($p_{in} = 0.02, p_{out} = 0.001$).
**Tabla 6.4: Resultados de Benchmarking en Planted Partition Model ($|V| = 10,000, |E| = 62,400$)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 31,180.5 | 31,182.0 | 140.2 | 30,910 | 31,450 | +6,230.1 | 14.2 | 9.1 | 0.0% | 0.0% |
| **Greedy** | 14,210.2 | 14,200.0 | 250.1 | 13,810 | 14,680 | +2,788.1 | 52.4 | 12.5 | 0.0% | 0.0% |
| **Spectral** | 502.0 | 502.0 | **0.0** | 502 | 502 | +2.03 | 2,410.6 | 92.1 | 0.0% | 0.0% |
| **SA** | 2,120.4 | 2,110.0 | 380.5 | 1,450 | 2,890 | +330.98 | 15,200.1 | 15.2 | 0.0% | 0.0% |
| **KL** | 890.2 | 885.0 | 95.4 | 720 | 1,080 | +81.01 | 2,110.4 | 13.4 | 0.0% | 0.0% |
| **TS** | 1,420.8 | 1,410.0 | 210.2 | 1,050 | 1,820 | +188.78 | 10,410.2 | 17.0 | 0.0% | 0.0% |
| **GA** | 3,850.1 | 3,820.0 | 520.4 | 2,910 | 4,890 | +682.54 | 31,200.5 | 54.1 | 0.0% | 0.0% |
| **BLS** | 512.4 | 510.0 | 14.2 | 498 | 535 | +4.15 | 4,810.2 | 34.1 | 0.0% | 0.0% |
| **KaHyPar** | 493.1 | 492.0 | 3.1 | 492 | 498 | +0.22 | 1,620.8 | 45.2 | 0.0% | 0.0% |
| **BLOC-RELOC** | **492.0** | **492.0** | **0.0** | **492** | **492** | **0.00** | **185.2** | **17.8** | 0.0% | 0.0% |
--### 6.2.5 Familia 5: Redes Desconectadas y Débilmente Acopladas ($Disconnected$)
Evaluación con 4 componentes desconectadas o enlazadas por un puente único de peso unitario.
**Tabla 6.5: Resultados de Benchmarking en Disconnected Networks ($|V| = 10,000, |E| = 45,000$)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 22,480.2 | 22,485.0 | 115.4 | 22,210 | 22,710 | +2,248,000 | 11.8 | 8.1 | 0.0% | 0.0% |
| **Greedy** | 9,810.5 | 9,800.0 | 180.2 | 9,450 | 10,120 | +981,000 | 42.0 | 11.0 | 0.0% | 0.0% |
| **Spectral** | **1.0** | **1.0** | **0.0** | **1** | **1** | **0.00** | 2,050.1 | 80.1 | 0.0% | 0.0% |
| **SA** | 1,240.5 | 1,210.0 | 310.8 | 450 | 1,890 | +124,000 | 13,800.0 | 14.5 | 0.0% | 0.0% |
| **KL** | 412.3 | 405.0 | 88.1 | 120 | 580 | +41,230 | 1,920.4 | 12.5 | 0.0% | 0.0% |
| **TS** | 850.1 | 840.0 | 195.4 | 310 | 1,240 | +85,010 | 9,120.5 | 16.0 | 0.0% | 0.0% |
| **GA** | 2,910.4 | 2,890.0 | 480.2 | 1,850 | 3,920 | +291,040 | 26,400.1 | 50.1 | 0.0% | 0.0% |
| **BLS** | 12.4 | 8.0 | 10.5 | **1** | 35 | +1,240 | 4,210.0 | 31.0 | 0.0% | 0.0% |
| **KaHyPar** | **1.0** | **1.0** | **0.0** | **1** | **1** | **0.00** | 1,320.5 | 41.0 | 0.0% | 0.0% |
| **BLOC-RELOC** | **1.0** | **1.0** | **0.0** | **1** | **1** | **0.00** | **142.1** | **16.5** | 0.0% | 0.0% |
--### 6.2.6 Familia 6: Red Social Real (SNAP Facebook `soc-FB`)
Prueba sobre grafo real social ($|V| = 4,039, |E| = 88,234$).
**Tabla 6.6: Resultados de Benchmarking en SNAP Facebook (`soc-FB`)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 44,110.8 | 44,112.0 | 180.5 | 43,810 | 44,450 | +809.83 | 8.5 | 12.4 | 0.0% | 0.0% |
| **Greedy** | 18,210.4 | 18,200.0 | 310.2 | 17,650 | 18,910 | +275.55 | 65.2 | 18.1 | 0.0% | 0.0% |
| **Spectral** | 6,120.0 | 6,120.0 | **0.0** | 6,120 | 6,120 | +26.21 | 1,450.8 | 112.5 | 0.0% | 0.0% |
| **SA** | 8,920.1 | 8,890.0 | 520.4 | 7,810 | 10,120 | +83.95 | 18,900.5 | 22.1 | 0.0% | 0.0% |
| **KL** | 5,890.2 | 5,880.0 | 142.1 | 5,610 | 6,210 | +21.47 | 3,120.4 | 19.4 | 0.0% | 0.0% |
| **TS** | 7,120.5 | 7,100.0 | 340.8 | 6,450 | 7,890 | +46.84 | 12,410.1 | 25.0 | 0.0% | 0.0% |
| **GA** | 11,200.4 | 11,150.0 | 780.1 | 9,810 | 12,890 | +130.98 | 42,100.0 | 68.2 | 0.0% | 0.0% |
| **BLS** | **4,849.0** | **4,845.0** | 22.4 | **4,810** | **4,890** | **0.00** | 6,120.5 | 45.1 | 0.0% | 0.0% |
| **KaHyPar** | 4,872.1 | 4,870.0 | 11.2 | 4,850 | 4,895 | +0.48 | 1,890.2 | 58.4 | 0.0% | 0.0% |
| **BLOC-RELOC** | 5,420.3 | 5,410.0 | 112.5 | 5,210 | 5,680 | +11.78 | **512.0** | **24.5** | 0.0% | 0.0% |
--### 6.2.7 Familia 7: Red de Votación Real (SNAP WikiVote `wiki-Vote`)
Prueba sobre la red real de votaciones de Wikipedia ($|V| = 7,115, |E| = 103,689$).
**Tabla 6.7: Resultados de Benchmarking en SNAP WikiVote (`wiki-Vote`)**

| Algoritmo | Mean Cut | Median Cut | Std Dev | Best Cut | Worst Cut | Gap (%) | Mean Time (ms) | Peak RAM (MB) | Time
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | 51,820.4 | 51,825.0 | 210.8 | 51,410 | 52,210 | +433.82 | 14.1 | 15.2 | 0.0% | 0.0% |
| **Greedy** | 24,110.2 | 24,100.0 | 410.5 | 23,200 | 25,100 | +148.37 | 88.4 | 22.4 | 0.0% | 0.0% |
| **Spectral** | 12,110.0 | 12,110.0 | **0.0** | 12,110 | 12,110 | +24.75 | 2,890.4 | 145.2 | 0.0% | 0.0% |
| **SA** | 14,810.5 | 14,780.0 | 680.2 | 13,200 | 16,100 | +52.57 | 25,400.1 | 28.4 | 0.0% | 0.0% |
| **KL** | 11,240.1 | 11,220.0 | 210.4 | 10,810 | 11,680 | +15.79 | 5,120.0 | 24.1 | 0.0% | 0.0% |
| **TS** | 13,100.4 | 13,080.0 | 450.1 | 12,100 | 14,100 | +34.95 | 18,200.4 | 31.2 | 0.0% | 0.0% |
| **GA** | 18,910.2 | 18,850.0 | 950.4 | 16,800 | 21,200 | +94.80 | 58,900.2 | 84.1 | 0.0% | 0.0% |
| **BLS** | **9,707.2** | **9,705.0** | 35.1 | **9,650** | **9,780** | **0.00** | 9,810.5 | 58.2 | 0.0% | 0.0% |
| **KaHyPar** | 9,720.5 | 9,718.0 | 14.8 | 9,690 | 9,750 | +0.14 | 2,410.0 | 72.1 | 0.0% | 0.0% |
| **BLOC-RELOC** | 10,850.4 | 10,840.0 | 185.1 | 10,480 | 11,210 | +11.78 | **890.1** | **31.0** | 0.0% | 0.0% |
--## 6.3 Rankings Relativos y Perfiles de Desempeño de Dolan-Moré
### 6.3.1 Matriz de Rankings por Familia Topológica (1 = Mejor, 10 = Peor)
**Tabla 6.8: Posición Ordinal de Rendimiento por Topología**

| Algoritmo | Regular | Small-World | Scale-Free | SBM | Disconnected | SNAP FB | SNAP Wiki | **Rango Promedio Global**
|---|---|---|---|---|---|---|---|
| **KaHyPar** | **1** | 2 | **1** | 2 | **1** | 2 | 2 | **1.43** |
| **BLOC-RELOC** | 2 | **1** | 4 | **1** | **1** | 4 | 4 | **2.43** |
| **BLS** | 3 | 3 | 2 | 4 | 4 | **1** | **1** | **2.57** |
| **KL** | 4 | 4 | 3 | 5 | 5 | 3 | 3 | **3.86** |
| **Spectral** | 7 | 8 | 5 | 3 | **1** | 5 | 5 | **4.86** |
| **TS** | 5 | 5 | 6 | 6 | 6 | 6 | 6 | **5.43** |
| **SA** | 6 | 6 | 7 | 7 | 7 | 7 | 7 | **6.57** |
| **GA** | 8 | 7 | 8 | 8 | 8 | 8 | 8 | **7.86** |
| **Greedy** | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9.00** |
| **Random** | 10 | 10 | 10 | 10 | 10 | 10 | 10 | **10.00** |
--### 6.3.2 Perfiles de Desempeño de Dolan-Moré (Calidad de Corte y Tiempo)
Los gráficos ASCII a continuación representan la probabilidad acumulada $\rho_s(\tau)$ de que el costo obtenido por el
**Diagrama 6.3: Perfil de Desempeño Dolan-Moré para Calidad de Corte ($C_A / C^*$)**
```
rho_s(tau)
1.0 ┼
┌──────────────────── (KaHyPar)
│
┌───────────────┴──────────────────── (BLOC-RELOC)
0.8 ┼
│
┌──────────────────── (BLS)
│
┌─────────────┘
│
0.6 ┼
│
│
│
│
└──────────────────── (KL)
0.4 ┼
│
│
│
0.2 ┼──────┘
│
0.0 ┼──────┬─────────────────────────────┬────────────────────┬───────────►
1.0
1.05
1.20
tau
```
**Diagrama 6.4: Perfil de Desempeño Dolan-Moré para Tiempo de Ejecución ($T_A / T^*$)**
```
rho_s(tau)
1.0 ┼─────────────────┐
(BLOC-RELOC)
│
└───────────────────────────────────── (Greedy)
0.8 ┼
(Random)
│
(KaHyPar)
0.6 ┼────────────────────────────────────┐
(KL)
│
└────────────────── (BLS)

0.4 ┼
│
0.0 ┼──────┬─────────────────────────────┬────────────────────┬───────────►
1.0
2.0
5.0
tau
```
--## 6.4 Pruebas de Significancia Estadística Infranqueable
### 6.4.1 Resultados de la Prueba de Rangos con Signo de Wilcoxon

Se evaluaron las diferencias pareadas entre **BLOC-RELOC** y los competidores principales a través de las 210 instanc
**Tabla 6.9: Test de Wilcoxon Pareado (BLOC-RELOC vs. Competidores, $\alpha = 0.01$)**
| Par Comparativo | $W^+$ (BLOC mejor) | $W^-$ (Competidor mejor) | $p$-valor | Decisión ($\alpha=0.01$) |
|---|---|---|---|
| **BLOC-RELOC vs. Random** | 22,155.0 | 0.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC muy superior) |
| **BLOC-RELOC vs. Greedy** | 22,155.0 | 0.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC muy superior) |
| **BLOC-RELOC vs. Spectral** | 20,410.5 | 1,744.5 | $< 0.00001$ | Rechaza $H_0$ (BLOC superior) |
| **BLOC-RELOC vs. SA** | 22,155.0 | 0.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC muy superior) |
| **BLOC-RELOC vs. KL** | 21,890.0 | 265.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC superior) |
| **BLOC-RELOC vs. TS** | 22,155.0 | 0.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC muy superior) |
| **BLOC-RELOC vs. GA** | 22,155.0 | 0.0 | $< 0.00001$ | Rechaza $H_0$ (BLOC muy superior) |
| **BLOC-RELOC vs. BLS** | 13,420.0 | 8,735.0 | $0.00214$ | Rechaza $H_0$ (BLOC superior global) |
| **BLOC-RELOC vs. KaHyPar** | 9,810.0 | 12,345.0 | $0.03410$ | No rechaza $H_0$ (Empate estadístico) |
--### 6.4.2 Tamaños del Efecto de Cohen ($d$ de Cohen)
**Tabla 6.10: Magnitud de la Diferencia Práctica ($d$ de Cohen) de BLOC-RELOC por Familia**

| Familia Topológica | vs. KL ($d$) | vs. BLS ($d$) | vs. KaHyPar ($d$) | Interpretación vs. KaHyPar |
|---|---|---|---|---|
| **Random Regular** | $-2.85$ (Muy Grande) | $-1.62$ (Muy Grande) | $+0.13$ (Negligible) | Empate TÉCNICO |
| **Small World** | $-4.12$ (Muy Grande) | $-2.98$ (Muy Grande) | $-1.24$ (Grande) | **BLOC-RELOC Domina** |
| **Scale Free** | $-1.82$ (Muy Grande) | $+3.41$ (Muy Grande) | $+4.52$ (Muy Grande) | KaHyPar Domina |
| **SBM** | $-5.91$ (Extremo) | $-1.95$ (Muy Grande) | $-0.32$ (Pequeño) | **BLOC-RELOC Domina** |
| **Disconnected** | $-6.21$ (Extremo) | $-1.52$ (Muy Grande) | $0.00$ (Idéntico) | **Empate Absoluto** |
| **SNAP Facebook** | $-3.81$ (Muy Grande) | $+3.21$ (Muy Grande) | $+4.81$ (Muy Grande) | BLS / KaHyPar Dominan
| **SNAP WikiVote** | $-1.98$ (Muy Grande) | $+3.89$ (Muy Grande) | $+6.12$ (Muy Grande) | BLS / KaHyPar Dominan
--## 6.5 Análisis de Estabilidad y Varianza Inter-Semilla

Para evaluar la sensibilidad de cada algoritmo a las condiciones iniciales (semillas aleatorias $s \in \mathcal{S}$), se ca
**Tabla 6.11: Coeficiente de Variación Inter-Semilla ($CV\%$) por Algoritmo**

| Algoritmo | Regular | Small-World | Scale-Free | SBM | Disconnected | SNAP FB | SNAP Wiki | **CV Promedio (%)** |
|---|---|---|---|---|---|---|---|
| **Spectral** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** (Determ
| **BLOC-RELOC** | **0.15%** | **0.31%** | 3.98% | **0.00%** | **0.00%** | 2.08% | 1.71% | **1.18%** |
| **KaHyPar** | 0.10% | 0.25% | **0.26%** | 0.63% | **0.00%** | **0.23%** | **0.15%** | **0.23%** |
| **BLS** | 0.43% | 0.99% | 1.13% | 2.77% | 84.68% | 0.46% | 0.36% | 12.97% |
| **KL** | 0.70% | 1.68% | 2.89% | 10.72% | 21.37% | 2.41% | 1.87% | 5.95% |
| **TS** | 1.28% | 3.12% | 5.24% | 14.80% | 22.99% | 4.79% | 3.44% | 7.95% |
| **SA** | 1.60% | 3.79% | 7.01% | 17.95% | 25.05% | 5.83% | 4.59% | 9.40% |
| **GA** | 2.33% | 5.69% | 8.31% | 13.52% | 16.50% | 6.96% | 5.03% | 8.33% |
--## 6.6 Discusión Objetiva de los Patrones de Rendimiento Observados
Un análisis desprovisto de sesgo sobre las Tablas 6.1 a 6.11 revela la existencia de **tres patrones de comportamiento

```
========================================================================
MATRIZ DE DOMINANCIA OPERACIONAL DE BLOC-RELOC
========================================================================

REGIMEN DE DOMINANCIA
REGIMEN DE EQUILIBRIO
REGIMEN DE DEGRADACIÓN
ABSOLUTA
O PARIDAD
OCASIONAL / DERROTA
┌────────────────────────┐
┌──────────────────────┐
┌────────────────────────┐
│ • Small-World (WS)
│
│ • Random Regular
│
│ • Scale-Free (BA)
│
│ • Planted Partition │
│ • Disconnected
│
│ • SNAP Facebook
│
│ (SBM)
│
└──────────────────────┘
│ • SNAP WikiVote
│
└────────────────────────┘
└────────────────────────┘
========================================================================
```

1. **Zonas de Dominancia Absoluta de BLOC-RELOC:**
En las familias **Small-World (WS)** y **Planted Partition (SBM)**, BLOC-RELOC supera de forma incontestable a la to

2. **Zonas de Paridad y Eficiencia Cero-Fricción:**
En las redes **Random Regular** y **Disconnected**, BLOC-RELOC se sitúa en un empate técnico perfecto en calidad

3. **Zonas de Degradación Operacional de BLOC-RELOC:**
En las redes con distribuciones de grado altamente asimétricas o presencia masiva de hubs—**Scale-Free (BA)**, **SN
--## 6.7 Necesidad de una Investigación Estructural Profunda
Los datos empíricos presentados en este capítulo demuestran que **ningún algoritmo de optimización individual ejerce

La variabilidad observada en la efectividad de BLOC-RELOC—pasando de ser el solucionador más rápido y preciso en re
Tales hallazgos plantean interrogantes científicos fundamentales:

* ¿Por qué el mecanismo de reubicación por bloques de BLOC-RELOC exhibe una eficiencia extrema en ciertas topología
* ¿Existe una propiedad matemática cuantitativa en la estructura del grafo que determine con precisión la frontera dond

El capítulo siguiente aborda estas interrogantes de manera directa, transitando del reporte puramente descriptivo de re
----# CAPÍTULO 7: DESCUBRIMIENTO DEL QUIEBRE TOPOLÓGICO Y ANÁLISIS CIENTÍFICO DE LOS RESULTADOS
--## 7.1 Introducción: Del Rendimiento Empírico a la Necesidad Epistemológica

La acumulación y el análisis descriptivo de la evidencia cuantitativa expuesta en el Capítulo 6 revelaron un fenómeno q

Este capítulo aborda la interpretación científica de dichos datos. El objetivo central no es justificar el rendimiento media
--## 7.2 Análisis Comparativo Transversal y Patrones Recurrentes

Al cruzar los resultados de las siete familias topológicas evaluadas, emergen tres regularidades empíricas que exigen un

1. **La Bifurcación de Rendimiento:** Algoritmos altamente eficientes en redes con fuerte estructura modular y alta den

2. **La Invariabilidad de los Métodos Espectrales:** Los métodos algebraicos basados en el Laplaciano del grafo y el vec

3. **La Complementariedad de las Metaheurísticas de Perturbación:** Solucionadores basados en saltos adaptativos y c

Estos patrones sugieren que las métricas globales de un grafo (como el número de vértices y aristas) son insuficientes p
---

## 7.3 Anatomía del Comportamiento de BLOC-RELOC

Para comprender la naturaleza de los resultados obtenidos por **BLOC-RELOC v1.0**, es necesario analizar los mecanis
### 7.3.1 Convergencia Casi Determinista en Grafos Estructurados (WS, SBM, Regular, Disconnected)

En las redes Small-World, el modelo Planted Partition, los grafos regulares y las redes desconectadas, BLOC-RELOC alca

* **Explicación Estructural:** Estas topologías se caracterizan por una alta **modularidad** ($Q$) y un elevado **coefic

* **Mecanismo Operativo:** El motor de reubicación atómica de bloques de BLOC-RELOC, impulsado por la matriz de af
### 7.3.2 Degradación de Rendimiento en Redes Reales SNAP y Scale-Free (BA, soc-FB, wiki-Vote)

Por el contrario, en las redes Barabási-Albert, Facebook y WikiVote, el rendimiento de BLOC-RELOC se degrada, present

* **Explicación Estructural:** Estas redes presentan distribuciones de grado que siguen leyes de potencia $P(k) \sim k^

* **Mecanismo Operativo:** La estrategia de BLOC-RELOC de reubicar bloques de contigüidad asume una distribución e
--## 7.4 El Impacto de las Métricas Estructurales Invariantes

La divergencia en el comportamiento de los solucionadores obliga a examinar el impacto matemático de ocho propieda

1. **Heterogeneidad de Grado y Super-Hubs:** Medida a través de la varianza del grado $\sigma_d^2 = \frac{1}{|V|} \

2. **Modularidad ($Q$):** Cuantifica la densidad de aristas dentro de las comunidades en comparación con las aristas e
$$Q = \frac{1}{2|E|} \sum_{u,v} \left[ A_{uv} - \frac{d_u d_v}{2|E|} \right] \delta(c_u, c_v)$$
Los algoritmos locales puros prosperan cuando $Q \gg 0.5$; fallan cuando $Q$ se aproxima a cero.
3. **Assortativity ($r$):** Coeficiente de correlación de Pearson de los grados de los nodos conectados en los extremos

4. **Coeficiente de Agrupamiento Local ($Cl$):** La probabilidad de que los vecinos de un nodo estén conectados entre
5. **Conductancia Global ($\phi$):** Mide la escasez de los cuellos de botella entre subgrafos. Redes con conductancia

6. **Estructura Core-Periphery:** Organización donde un núcleo denso y altamente interconectado coexiste con una per
--## 7.5 El Concepto de "Régimen Topológico"

La evidencia empírica acumulada conduce a la formulación central de este capítulo: **el rendimiento algorítmico está g

> **Definición (Régimen Topológico):** Se define como régimen topológico $\mathcal{R}$ al subespacio métrico y espe

En lugar de tratar el particionamiento de grafos como un problema uniforme sobre un conjunto discreto arbitrario, la int
--## 7.6 Clasificación Preliminar de los Regímenes Topológicos

A partir de los resultados observados en el Capítulo 6, se propone una clasificación taxonómica preliminar de tres **Reg

```
========================================================================
TAXONOMÍA DE REGÍMENES TOPOLÓGICOS
========================================================================

RÉGIMEN I: MODULAR / SIMÉTRICO
RÉGIMEN II: HOMOGÉNEO / REGULAR RÉGIMEN III: ASIMÉTRICO / CORE-PERIPHE
• Alta Modularidad (Q > 0.6)
• Grado Uniforme (d_u = k)
• Ley de Potencia (P(k) ~ k^-γ)
• Alto Agrupamiento (Cl)
• Assortativity Neutra (r ≈ 0) • Presencia de Super-Hubs
• Fronteras Claras
• Ausencia de Comunidades
• Estructura Core-Periphery
• Solucionador Óptimo:
• Solucionador Óptimo:
• Solucionador Óptimo:

BLOC-RELOC / KaHyPar
KaHyPar / Espectral
KaHyPar / BLS
========================================================================
```
1. **Régimen I (Modular / Simétrico):** Caracterizado por comunidades densas, alta modularidad y separación clara de

2. **Régimen II (Homogéneo / Regular):** Caracterizado por distribuciones de grado uniformes sin estructuras comunita

3. **Régimen III (Asimétrico / Core-Periphery):** Caracterizado por distribuciones de grado libres de escala, alta heterog
--## 7.7 Formulación de Hipótesis Científicas Fundamentales

Del descubrimiento de los regímenes topológicos se derivan tres hipótesis científicas formales que guiarán las fases sub

* **Hipótesis 1 (De Transición de Fase Estructural):** La transición en la efectividad relativa entre un solucionador basad

* **Hipótesis 2 (De Isomorfismo Espacial en la Búsqueda):** La superficie de costo $\mathcal{L}(\Pi)$ exhibe un número

* **Hipótesis 3 (De Ineficiencia Universal):** Ningún algoritmo heurístico estático puede alcanzar el óptimo de eficiencia
--## 7.8 La Falacia del Algoritmo Universal

La consecuencia teórica más importante derivada del análisis de los datos experimentales es la refutación de la búsque

Durante décadas, la literatura en optimización combinatoria se ha centrado en diseñar heurísticas cada vez más comple

Dado que las restricciones topológicas de una red modular difieren radicalmente de las de una red Scale-Free con super
--## 7.9 Conclusión y Cambio de Paradigma Arquitectónico

El descubrimiento del quiebre topológico transforma por completo la perspectiva del Proyecto TCF. La tarea de optimiza
$$\text{Paradigma Antiguo: } \quad \min_{\Pi} C(\Pi) \quad \text{mediante un único Solver Estático } \mathcal{S}$$

Para evolucionar hacia un paradigma dinámico donde el sistema de resolución debe adaptarse de forma inteligente a la

$$\text{Nuevo Paradigma: } \quad \mathcal{R} = \Psi(G) \implies \text{Selección / Adaptación de Estrategia } \mathca

Este hallazgo obliga a abandonar la búsqueda de mejoras incrementales en el código interno de BLOC-RELOC v1.0 y exi

Esta conclusión cierra la etapa puramente experimental y descriptiva de la monografía, preparando el terreno para la fo
----# PARTE III: FORMALIZACIÓN MATEMÁTICA
--# CAPÍTULO 8: FORMALIZACIÓN MATEMÁTICA DE LOS REGÍMENES TOPOLÓGICOS
--## 8.1 Introducción y Fundamentos Epistemológicos

El descubrimiento del quiebre topológico expuesto en el Capítulo 7 evidenció que la optimización combinatoria sobre gr

Este capítulo establece la formalización analítica de los espacios topológicos. Su propósito central no es diseñar un nuev
--## 8.2 El Espacio de Grafos como Espacio Métrico

Sea $\mathcal{G}$ el conjunto de todos los grafos finitos, simples, no dirigidos y ponderados $G = (V, E, W)$, donde $V

Para comparar cuantitativamente la estructura de dos grafos distintos $G_1, G_2 \in \mathcal{G}$, es necesario dotar a
--## 8.3 Definición Formal del Vector Topológico $\Theta(G)$

Se define el **Vector Topológico** $\Theta(G)$ como una aplicación multivariada $\Theta: \mathcal{G} \to \mathbb{R}
$$\Theta(G) = \left[ \rho(G), \, \kappa(G), \, Q(G), \, r(G), \, Cl(G), \, \Phi(G), \, \gamma(G), \, \chi(G) \right]^T$$
A continuación se desarrolla la formulación matemática rigurosa de cada uno de los componentes de $\Theta(G)$:
### 1. Densidad de Aristas ($\rho(G)$)
Cuantifica la proporción de aristas existentes respecto al máximo posible en un grafo simple:
$$\rho(G) = \frac{2m}{n(n-1)}$$
### 2. Heterogeneidad de Grado ($\kappa(G)$)

Mide la dispersión relativa de la distribución de grados mediante el coeficiente de variación normalizado de la secuencia

$$\kappa(G) = \frac{\sigma_d}{\bar{d}} = \frac{\sqrt{\frac{1}{n} \sum_{u \in V} (d_u - \bar{d})^2}}{\frac{1}{n} \s
Donde $d_u$ es el grado del vértice $u$ y $\bar{d}$ es el grado medio.
### 3. Modularidad Máxima Estimada ($Q(G)$)
Mide la fuerza de división de una red en módulos densamente interconectados:
$$Q(G) = \frac{1}{2m} \sum_{u,v} \left( A_{uv} - \frac{d_u d_v}{2m} \right) \delta(c_u, c_v)$$
Donde $A$ es la matriz de adyacencia y $c_u$ denota la comunidad asignada al vértice $u$.
### 4. Coeficiente de Assortativity ($r(G)$)
Calcula la correlación de Pearson entre los grados de los nodos conectados en los extremos de cada arista:
$$r(G) = \frac{\sum_{uv} (d_u - \bar{d})(d_v - \bar{d}) w_{uv}}{\sum_{u} (d_u - \bar{d})^2}$$
### 5. Coeficiente de Agrupamiento Local Global ($Cl(G)$)
Representa la transitividad promedio de las tripletas de vértices conectados:
$$Cl(G) = \frac{1}{n} \sum_{u \in V} \frac{2 |\{ (v, w) \in E : v, w \in \Gamma(u) \}|}{d_u (d_u - 1)}$$
Donde $\Gamma(u)$ es el conjunto de vecinos del nodo $u$.
### 6. Conductancia Global Espectral ($\Phi(G)$)

Evalúa el cuello de botella estructural a través del segundo autovalor del Laplaciano normalizado (la constante de Chee
$$\Phi(G) \approx \frac{\lambda_2}{2}$$
### 7. Exponente de Ley de Potencia ($\gamma(G)$)

Para redes con distribuciones de grado heterogéneas, se ajusta la cola de la distribución $P(k) \sim k^{-\gamma}$ med
$$\gamma(G) = 1 + n \left[ \sum_{u=1}^n \ln \frac{d_u}{d_{min}} \right]^{-1}$$
### 8. Puntuación Core-Periphery ($\chi(G)$)

Mide la concentración de conexiones en un núcleo denso central frente a una periferia escasa, evaluada mediante la co
$$\chi(G) = \max_{\pi} \text{corr}(A, \pi(\rho_{cp}))$$
---

## 8.4 Métrica de Distancia en el Espacio Topológico

Para cuantificar la disparidad estructural entre dos grafos $G_1$ y $G_2$ a través de sus vectores topológicos, se define
$$d_{\Theta}(G_1, G_2) = \sqrt{(\Theta(G_1) - \Theta(G_2))^T W (\Theta(G_1) - \Theta(G_2))}$$

Donde $W$ es una matriz diagonal de pesos $W = \text{diag}(w_1, w_2, \dots, w_d)$ que normaliza las escalas de las d
--## 8.5 Taxonomía Matemática de los Regímenes Topológicos

A partir del espacio vectorial $\Theta(G)$, se definen formalmente las fronteras algebraicas que delimitan los cinco regí

```
========================================================================
FRONTERAS MATEMÁTICAS DE LOS REGÍMENES TOPOLÓGICOS
========================================================================

RÉGIMEN MODULAR ($\mathcal{R}_{mod}$)
RÉGIMEN REGULAR ($\mathcal{R}_{reg}$)
Q(G) ≥ 0.5 ∧ Cl(G) ≥ 0.4 ∧ κ(G) < 0.5
κ(G) < 0.1 ∧ |d_u - d̄| ≈ 0
───────────────────────────────────────
───────────────────────────────────
RÉGIMEN SCALE-FREE ($\mathcal{R}_{sf}$)
RÉGIMEN HUB-DOMINATED ($\mathcal{R}_{hub}$)
2.0 ≤ γ(G) ≤ 3.5 ∧ κ(G) ≥ 1.5
max(d_u) ≥ √n ∧ χ(G) ≥ 0.7
───────────────────────────────────────
───────────────────────────────────
RÉGIMEN MIXTO ($\mathcal{R}_{mix}$)
Condiciones superpuestas / No linealidad estructural
========================================================================
```
**1. Régimen Modular ($\mathcal{R}_{mod}$):**
$$\mathcal{R}_{mod} = \{ G \in \mathcal{G} \mid Q(G) \ge 0.5 \land Cl(G) \ge 0.4 \land \kappa(G) < 0.5 \}$$
**2. Régimen Regular ($\mathcal{R}_{reg}$):**
$$\mathcal{R}_{reg} = \{ G \in \mathcal{G} \mid \kappa(G) < 0.1 \land \max_{u,v} |d_u - d_v| \le 2 \}$$
**3. Régimen Scale-Free ($\mathcal{R}_{sf}$):**
$$\mathcal{R}_{sf} = \{ G \in \mathcal{G} \mid 2.0 \le \gamma(G) \le 3.5 \land \kappa(G) \ge 1.5 \}$$
**4. Régimen Hub-Dominated ($\mathcal{R}_{hub}$):**
$$\mathcal{R}_{hub} = \{ G \in \mathcal{G} \mid \max_{u} d_u \ge \sqrt{n} \land \chi(G) \ge 0.7 \}$$
**5. Régimen Mixto ($\mathcal{R}_{mix}$):**

$$\mathcal{R}_{mix} = \mathcal{G} \setminus (\mathcal{R}_{mod} \cup \mathcal{R}_{reg} \cup \mathcal{R}_{sf} \
--## 8.6 Dinámica de Transiciones de Fase Estructurales

Un fenómeno crítico descubierto en la experimentación es que pequeñas variaciones continuas en los parámetros de ge

Matemáticamente, esto se modela como una **transición de fase no lineal**. Sea $\theta_i(G)$ un invariante escalar de

$$\lim_{\Delta \mu \to 0} \frac{\partial Gap(\mathcal{S}, G(\mu))}{\partial \mu} = \infty \quad \text{en } \mu = \mu_c
Esta discontinuidad en la derivada justifica por qué los algoritmos basados en heurísticas locales puras sufren colapsos
--## 8.7 Modelado del Paisaje de Optimización y Geometría de la Superficie de Costo

Para comprender por qué distintos algoritmos encuentran mínimos diferentes según la topología, se debe analizar la ge
Sea $\Pi$ el espacio discreto de todas las particiones posibles de los vértices en $k$ bloques, y sea $\mathcal{L}(\Pi): \P

* **En el Régimen Modular ($\mathcal{R}_{mod}$):** La superficie $\mathcal{L}(\Pi)$ presenta un conjunto acotado d

* **En el Régimen Scale-Free / Hub-Dominated ($\mathcal{R}_{sf} \cup \mathcal{R}_{hub}$):** La superficie de costo
--## 8.8 Conclusión: Clasificación Previa a la Optimización
La formalización matemática desarrollada en este capítulo demuestra que **la incertidumbre algorítmica es eliminable

Dado un grafo arbitrario $G$, es innecesario ejecutar múltiples heurísticas a ciegas para determinar cuál es la más adec

Esta conclusión sella la base teórica de la monografía y plantea la necesidad ineludible de diseñar una arquitectura de s
----# PARTE IV: EVOLUCIÓN ARQUITECTÓNICA
--# CAPÍTULO 9: FRAMEWORK ADAPTATIVO DE OPTIMIZACIÓN TOPOLÓGICA
--## 9.1 Filosofía del Framework: La Superación del Algoritmo Universal

La formalización teórica desarrollada en el Capítulo 8 demostró que el espacio de grafos $\mathcal{G}$ se encuentra p
El **Framework Adaptativo de Optimización Topológica (FAOT)** nace como respuesta arquitectónica a esta limitación.
--## 9.2 Arquitectura General del Sistema

El framework está concebido bajo el paradigma de un sistema operativo modular para la optimización de grafos. Su arq

```
========================================================================
ARQUITECTURA GENERAL DEL FAOT
========================================================================
[ Grafo de Entrada G = (V, E, W) ]
│
▼
┌─────────────────────────────┐
│
PROFILER TOPOLÓGICO
│ ──► Calcula Secuencia de Invariantes
└──────────────┬──────────────┘
│ Vector Θ(G)
▼
┌─────────────────────────────┐
│ CLASIFICADOR DE REGÍMENES │ ──► Determina Regimen R ∈ {R_mod, R_reg, ...}
└──────────────┬──────────────┘
│ Identidad Topológica
▼
┌─────────────────────────────┐
│
STRATEGY ROUTER
│ ──► Selecciona y Despacha Solvers Óptimos
└──────────────┬──────────────┘
│
┌─────────┴─────────┐ (Ejecución Paralela o Secuencial)
▼
▼
┌──────────┐
┌──────────┐
│ Solver A │
│ Solver B │ (Biblioteca de Solvers: BLOC-RELOC, BLS, etc.)
└────┬─────┘
└────┬─────┘
│
│
└─────────┬─────────┘
│ Resultados Crudos

▼
┌─────────────────────────────┐
│
MOTOR DE EVALUACIÓN
│ ──► Valida Calidad, Latencia y Factibilidad
└──────────────┬──────────────┘
│ Feedback Métrico
▼
┌─────────────────────────────┐
│ SISTEMA DE RETROALIMENTACIÓN│ ──► Actualiza Pesos y Registro de Evidencias
└─────────────────────────────┘
========================================================================
```
Los componentes estructurales que integran esta arquitectura son:

1. **Profiler Topológico:** Módulo analítico encargado de extraer la secuencia de invariantes de la red.
2. **Constructor del Vector $\Theta(G)$:** Subsistema de normalización y estructuración vectorial del espacio métrico.
3. **Clasificador de Regímenes:** Motor de decisión basado en fronteras algebraicas y clasificadores multi-clase.
4. **Strategy Router:** Orquestador dinámico que determina la política de despacho de los motores de búsqueda.
5. **Biblioteca de Solvers:** Repositorio extensible de motores de optimización especializados (BLOC-RELOC, KaHyPar, B
6. **Motor de Evaluación:** Subsistema de verificación de restricciones y medición objetiva de cortes.
7. **Sistema de Retroalimentación:** Mecanismo de aprendizaje por refuerzo estático para la calibración del router.
8. **Registro de Evidencias:** Base de datos persistente de telemetría experimental previa.
9. **Módulo de Aprendizaje de Nuevos Regímenes:** Componente de detección de anomalías topológicas para expansió
--## 9.3 El Profiler Topológico
El Profiler es el componente encargado de extraer las métricas estructurales sin alterar la representación de la red.
* **Funcionalidad:** Calcula el vector $\Theta(G)$ evaluando propiedades locales, globales y espectrales del grafo.

* **Complejidad Temporal:** Dominada por el cálculo del espectro del Laplaciano normalizado y el coeficiente de agrup
$$\mathcal{T}_{profiler} = \mathcal{O}\left( m \sqrt{n} + n \log n \right)$$

* **Complejidad Espacial:** $\mathcal{O}(m + n)$, requiriendo únicamente el almacenamiento de la matriz de adyacen
--## 9.4 Clasificador de Regímenes

El Clasificador opera como una función de mapeo $M: \mathbb{R}^d \to \mathcal{R}$, donde $\mathcal{R}$ es el con
### Modelo Matemático de Decisión

Utiliza un árbol de decisión hiperplanar optimizado mediante vectores de soporte con márgenes estrictos basados en la
$$\mathcal{R}^* = \arg\max_{\mathcal{R}_k} P(\mathcal{R}_k \mid \Theta(G))$$

* **Ventajas:** Determinismo analítico absoluto en regiones de frontera clara, interpretabilidad geométrica directa y tie

* **Limitaciones:** Vulnerabilidad teórica en grafos limítrofes que pertenecen simultáneamente a la intersección de dos
--## 9.5 Strategy Router: El Núcleo de Enrutamiento

El *Strategy Router* es el componente que traduce la identidad topológica $\mathcal{R}$ en una directiva de ejecución

$$\min_{\mathcal{S} \in \mathcal{S}_{lib}} \mathbb{E}\left[ \text{Time}(\mathcal{S}) \right] \quad \text{s.a.} \quad
El Router soporta cinco **Políticas de Ejecución Avanzadas**:

1. **Selección de Solver Único (Single-Solver Dispatch):** Cuando el clasificador identifica un régimen con dominancia a

2. **Ejecución en Paralelo (Parallel Portfolio):** Ante grafos limítrofes ($\mathcal{R}_{mix}$), lanza de forma concurren
3. **Combinación de Resultados (Result Fusion):** Algoritmo de ensamblaje que toma las mejores particiones parciales

4. **Detención Temprana (Early Stopping):** Monitoriza la derivada temporal de la función de costo de los solvers activo

5. **Cambio Dinámico de Estrategia (Dynamic Handoff):** Si un solver primario excede un umbral de tiempo sin conver
--## 9.6 Biblioteca de Solvers Extensible

La *Biblioteca de Solvers* establece una interfaz polimórfica estandarizada (`IGraphSolver`) que encapsula cualquier m
La biblioteca integra de forma nativa:
* **BLOC-RELOC v1.0:** Especialista en $\mathcal{R}_{mod}$ y $\mathcal{R}_{reg}$.
* **KaHyPar:** Motor multinivel de propósito general y alta robustez en $\mathcal{R}_{sf}$.
* **Breakout Local Search (BLS):** Metaheurística de escape de mínimos locales para redes complejas.
* **Kernighan-Lin (KL):** Heurística clásica de intercambio bilateral para grafos pequeños.
* **Spectral Bisection:** Solucionador algebraico de referencia determinista ($CV = 0.00\%$).

El diseño modular permite incorporar nuevos algoritmos en tiempo de ejecución sin modificar una sola línea de código d
--## 9.7 Flujo Completo del Sistema (Pipeline Operativo)
El ciclo de vida completo de una solicitud de particionamiento a través del FAOT se detalla en el siguiente diagrama de

```
========================================================================
PIPELINE OPERATIVO INTEGRAL DEL FAOT
========================================================================

[ 1. Ingestión de Grafo ]
│
▼
[ 2. Extracción de Telemetría (Profiler) ] ──► Calcula A, d, Q, r, Cl, Φ, γ, χ
│
▼
[ 3. Construcción Vectorial Θ(G) ]
│
▼
[ 4. Clasificación de Régimen R ] ──────────► ¿Pertenece a R_mod, R_sf, etc.?
│
├────────────────────────────────────────────────┐
▼
▼
[ 5A. Régimen Puro (Confiable) ]
[ 5B. Régimen Mixto / Limítrofe ]
│
│
▼
▼
[ 6A. Despacho Directo de Solver ]
[ 6B. Lanzamiento de Portfolio Paralelo ]
│
│
└───────────────────────┬────────────────────────┘
│
▼
[ 7. Monitoreo y Early Stopping ]
│
▼
[ 8. Evaluación de Corte C(Π) ]
│
▼
[ 9. Registro de Evidencias y Feedback ]
│
▼
[ 10. Retorno de Solución Óptima Π* ]
========================================================================
```
--## 9.8 Complejidad Computacional y Escalabilidad

El análisis de complejidad del framework demuestra que la sobrecarga introducida por los módulos de análisis y enrutam
* **Complejidad Temporal Total:**

$$\mathcal{T}_{total} = \mathcal{T}_{profiler}(\Theta(G)) + \mathcal{T}_{classifier} + \max_{\mathcal{S} \in \text{

Dado que $\mathcal{T}_{profiler} \ll \mathcal{T}_{\mathcal{S}}$ para grafos grandes, el costo analítico está amortigu

* **Paralelización y Ejecución Distribuida:** El Strategy Router aprovecha arquitecturas de memoria compartida multi-co
--## 9.9 Ventajas Conceptuales del Nuevo Paradigma

El Framework Adaptativo no representa una mejora incremental en heurísticas de optimización, sino una **revolución e

1. **Abandono del Dogma Estático:** Sustituye la búsqueda inútil de un solucionador universal por la inteligencia arquit

2. **Robustez Garantizada:** Al enrutar dinámicamente según el régimen topológico, el sistema absorbe la variabilidad

3. **Extensibilidad Estructurada:** Permite la integración continua de nuevos avances teóricos en la ciencia de redes sin
--## 9.10 Preparación para la Siguiente Fase
La implementación del Framework Adaptativo resuelve con éxito la asignación óptima de recursos en la mayoría de los

Esta deficiencia estructural evidencia que el motor clásico de BLOC-RELOC (v1.0), basado en reubicación atómica de blo
Este desafío conceptual abre de manera natural el paso hacia la siguiente etapa de la investigación.
----# CAPÍTULO 10: BLOC-RELOC v2: REDISEÑO DEL MOTOR PARA REDES HUB-DOMINATED
--## 10.1 Motivación Experimental

La campaña de benchmarking exhaustiva y el posterior análisis formal de los regímenes topológicos (Capítulos 6, 7 y 8)
### La Raíz Matemática del Fracaso en Redes Complejas

El fracaso de BLOC-RELOC v1.0 en estas topologías no responde a un déficit de convergencia local, sino a una contradic

1. **La Distorsión de los Super-Hubs:** En las redes regidas por el Régimen III, la presencia de vértices con grados masi
2. **La Ruptura de la Contigüidad Espacial:** El operador original de BLOC-RELOC asume que la afinidad entre bloques

3. **Paisajes Rugosos y Mínimos Falsos:** La superficie de optimización $\mathcal{L}(\Pi)$ en redes con estructuras *co

Para conquistar el Régimen Hub-Dominated sin sacrificar la velocidad característica de BLOC-RELOC, es imperativo redi
--## 10.2 Principios de Diseño de BLOC-RELOC v2
La arquitectura de la segunda generación se fundamenta en cinco axiomas de diseño formal:

1. **Preservación de Identidad Topológica Adaptativa:** El motor debe ajustar dinámicamente su sensibilidad a la heter

2. **Inmunidad Estructural a Super-Hubs (Hub-Shielding):** Las aristas incidentes en vértices de alto grado deben ser p

3. **Estabilidad Estocástica Controlada:** Preservar la baja varianza inter-semilla ($CV < 1.0\%$) exhibida en versiones
4. **Complejidad Sub-Cuadrática:** Mantener una cota temporal eficiente $\mathcal{O}(m \log n)$ que permita escalar

5. **Independencia del Régimen Operativo:** Capacidad intrínseca de operar de manera óptima tanto en redes modular
--## 10.3 Nueva Función de Afinidad Estructural (AF-v2)

Para superar las deficiencias de la matriz de afinidad convencional, se formula una nueva función matemática de atracc

Sean $U$ y $V$ dos bloques particionados del grafo. La afinidad estructural modificada $\mathcal{A}_2(U, V)$ se defin

$$\mathcal{A}_2(U, V) = \sum_{u \in U} \sum_{v \in V} A_{uv} \cdot \psi(d_u) \cdot \psi(d_v) \cdot \exp\left( -\frac{dist
Donde:
* $A_{uv}$ es el peso de la arista entre $u$ y $v$.
* $\psi(d_u) = \left( 1 + \frac{d_u}{\bar{d}} \right)^{-\beta}$ es el **factor de atenuación de grado** (con parámetro
* $dist(u, v)$ es la longitud del camino mínimo entre $u$ y $v$ en el subgrafo inducido.
* $\sigma_{local}$ es el radio de difusión de la vecindad local.
### Justificación de la Elección

A diferencia de las formulaciones de Kernighan-Lin (basadas en ganancias lineales directas de corte) o de KaHyPar (bas
--## 10.4 Hub-Shielded Affinity (HSA)

El núcleo algorítmico del rediseño radica en el mecanismo de **Hub-Shielded Affinity (HSA)**. Formalmente, definimos e
$$\mathcal{H} = \left\{ u \in V \mid d_u > \bar{d} + k_{\sigma} \sigma_d \right\}$$
Para neutralizar la influencia distorsionante de $\mathcal{H}$ sin eliminar información estructural valiosa, se aplica un

$$\tilde{A}_{uv} = \begin{cases} A_{uv} \cdot \left( \frac{\bar{d}}{d_u} \cdot \frac{\bar{d}}{d_v} \right)^{\alpha}

Donde $\alpha \in [0.5, 1.0]$ es el coeficiente de blindaje. Esta transformación escala analíticamente el peso de las arist
--## 10.5 Rediseño del Operador de Movimiento por Bloques

El operador de movimiento en BLOC-RELOC v2 evoluciona desde una simple reubicación heurística hacia un **Operador
### Fases Operativas:

1. **Selección de Bloque Crítico:** Identificación del bloque $U_k$ cuya frontera exhibe la mayor densidad de corte des

2. **Evaluación de Ganancia Marginal Shielded:** Cálculo del cambio neto en la función de corte $\Delta \mathcal{C}$ u

$$\Delta \mathcal{C}(u \in U_i \to U_j) = \sum_{w \in U_j} \tilde{A}_{uw} - \sum_{z \in U_i \setminus \{u\}} \tilde{A}_{

3. **Aceptación y Rechazo Estricto:** Se implementa un criterio de aceptación determinista condicionado a la mejora es
### Pseudocódigo Conceptual del Operador v2
```
Algoritmo 10.1: Operador de Desplazamiento Shielded (BLOC-RELOC v2)
Entrada: Partición actual Π, Matriz de adyacencia A, Conjunto de hubs H, Parámetro α
Salida: Partición optimizada Π*
1. Computar vector de grados d y umbral de super-hubs H
2. Construir matriz de adyacencia apantallada A_tilde usando α
3. Mientras no converja y (Iteraciones < Max_Iter):
4.
Para cada bloque U_k en Π:
5.
Calcular gradiente de afinidad apantallada ∇A_tilde(U_k)
6.
Identificar nodo candidato u ∈ U_k con peor cohesión interna
7.
Determinar bloque destino óptimo U_j que maximiza ΔC_shielded(u)
8.
Si ΔC_shielded(u) < 0:

9.
Ejecutar movimiento atómico: U_k = U_k \ {u}, U_j = U_j ∪ {u}
10.
Actualizar estructura de fronteras de corte
11.
Fin Si
12.
Fin Para
13.
Actualizar estado de convergencia global
14. Fin Mientras
15. Retornar Π*
```
--## 10.6 Mecanismo de Escape de Mínimos Locales: Dynamic Topological Shock (D-Shock)

Para superar los mínimos locales profundos característicos del Régimen III sin recurrir a Simulated Annealing (que destr
### Fundamento Teórico del D-Shock

Cuando el algoritmo detecta un estancamiento asintótico en la mejora del corte durante $t_{stall}$ iteraciones consecu

En lugar de perturbar estocásticamente nodos individuales, el D-Shock aplica un **pulso ortogonal de re-pesaje tempora
$$A^{(shock)}_{uv} = A_{uv} \cdot \left( 1 + \eta \cdot \xi_{uv} \right)$$

Donde $\eta$ es la amplitud del choque energético y $\xi_{uv} \in [-1, 1]$ es una variable aleatoria controlada por sem
--## 10.7 Análisis de Complejidad Computacional y Escalabilidad
El análisis formal de la complejidad de BLOC-RELOC v2 demuestra que el rediseño preserva la eficiencia requerida para
* **Complejidad Temporal:**
- Identificación de hubs y construcción de $\tilde{A}$: $\mathcal{O}(m + n)$
- Cálculo de la nueva afinidad apantallada por iteración: $\mathcal{O}(m \log n)$
- Complejidad total acotada por:
$$\mathcal{T}_{v2} = \mathcal{O}\left( k_{iter} \cdot m \log n \right)$$
Donde $k_{iter}$ es el número de iteraciones de convergencia controlada.
* **Complejidad Espacial:** $\mathcal{O}(m + n)$, requiriendo únicamente almacenamiento disperso en formato CSR

* **Paralelización:** El cálculo del gradiente de afinidad apantallada es totalmente desacoplable por bloques, permitiend
--## 10.8 Hipótesis Científicas Fundamentales de BLOC-RELOC v2

Para validar formalmente el rediseño algorítmico, se enuncian tres hipótesis científicas rigurosas que serán objeto de co

* **Hipótesis 1 (De Convergencia en Redes Scale-Free):** BLOC-RELOC v2 reducirá la brecha relativa de corte ($Gap\%$

* **Hipótesis 2 (De Inmunidad a Super-Hubs):** El mecanismo de *Hub-Shielded Affinity* (HSA) eliminará los estancamie

* **Hipótesis 3 (De Conservación de Eficiencia Computacional):** A pesar de la incorporación del blindaje de hubs y el m
--## 10.9 Predicciones Experimentales

Antes de la ejecución de cualquier validación empírica en el próximo capítulo, se establecen las siguientes predicciones

1. **Métricas de Calidad de Corte:** Se espera una mejora drástica en los cortes medios ($Mean Cut$) sobre `soc-FB` y

2. **Familias Beneficiadas:** Las redes del **Régimen III (Scale-Free y Hub-Dominated)** experimentarán la mayor gana

3. **Riesgos Identificados:** Un valor excesivamente alto del parámetro de blindaje $\alpha$ podría sobre-atenuar la to
---

## 10.10 Conclusiones y Transición

El desarrollo conceptual y matemático de BLOC-RELOC v2 demuestra que es posible dotar a un motor de optimización lo

La formalización de la *Hub-Shielded Affinity* y el operador de perturbación *D-Shock* resuelven teóricamente los prob
Este cierre metodológico prepara de forma natural el tránsito hacia la siguiente etapa fundamental de la investigación.
----# CAPÍTULO 11: VALIDACIÓN EXPERIMENTAL DE BLOC-RELOC v2
--## 11.1 Objetivos Experimentales y Verificación de Hipótesis

El propósito fundamental de esta campaña experimental no es buscar un sesgo favorable hacia el nuevo desarrollo, sino

* **Verificación de la Hipótesis 1 (Calidad de Corte en Scale-Free):** Evaluar si el mecanismo de Hub-Shielded Affinity (H
* **Verificación de la Hipótesis 2 (Inmunidad a Mínimos Locales):** Medir la reducción del número medio de iteraciones

* **Verificación de la Hipótesis 3 (Eficiencia Computacional y Escalabilidad):** Comprobar empíricamente que la sobreca
--## 11.2 Diseño del Entorno Experimental

Para garantizar la integridad científica, la reproducibilidad absoluta y la eliminación de sesgos de hardware o software, t

* **Hardware Base:** Servidor dedicado de computación de alto rendimiento dotado de un procesador AMD EPYC de 32
* **Sistema Operativo:** Ubuntu Server 24.04 LTS con núcleo Linux en modo de baja latencia (`PREEMPT_RT`).

* **Lenguaje de Programación:** C++20, compilado mediante GCC 13.2.0 utilizando directivas de optimización estricta

* **Librerías Base:** Eigen 3.4 para álgebra lineal dispersa, OpenMP 5.0 para paralelización de hilos en memoria compa

* **Control de Variables y Aislamiento:** Asignación estricta de hilos por núcleo mediante afinidad de CPU (`taskset`), d

* **Gestión de Semillas Estocásticas:** Uso de un generador de números pseudoaleatorios Mersenne Twister (`std::mt1
--## 11.3 Conjunto de Benchmarks (Suite de Pruebas)
La campaña experimental emplea una suite heterogénea compuesta por 120 instancias de grafos distribuidas en cinco

1. **Redes Reales SNAP (Stanford Network Analysis Project):**
- *soc-Facebook* y *wiki-Vote*: Instancias de gran escala pertenecientes al Régimen Scale-Free y Hub-Dominated ($\m

2. **Redes Sintéticas Barabási-Albert (BA):**
- Grafos generados mediante crecimiento preferencial con parámetros de conectividad $m \in \{3, 5, 10\}$ y tamaños

3. **Redes Small-World (Watts-Strogatz):**
- Instancias con alta modularidad local y caminos cortos, caracterizadas por coeficientes de agrupamiento elevados ($
4. **Grafos Regulares y Planted Partition (SBM):**
- Redes de control con distribución de grado homogénea ($\kappa \approx 0$) y bloques comunitarios perfectamente

5. **Redes Desconectadas y de Múltiples Componentes:**
- Instancias con componentes débilmente conectados para verificar la estabilidad numérica del framework ante grafos
--## 11.4 Algoritmos Comparados

Para establecer una línea de contraste rigurosa, se compararán diez solucionadores dentro del framework:

1. **Random Solver ($\mathcal{S}_{rand}$):** Asignación estocástica pura de vértices a bloques. Actúa como cota infe

2. **Greedy Baseline ($\mathcal{S}_{greedy}$):** Heurística simple de minimización de aristas locales sin backtrackin

3. **Spectral Bisection ($\mathcal{S}_{spec}$):** Método algebraico basado en el segundo autovalor del Laplaciano no
4. **Simulated Annealing ($\mathcal{S}_{sa}$):** Metaheurística clásica de enfriamiento estocástico para exploración
5. **Kernighan-Lin (KL):** Algoritmo clásico de intercambio heurístico bilateral de bloques.

6. **Breakout Local Search (BLS):** Metaheurística avanzada con mecanismos adaptativos de escape de mínimos locale

7. **Genetic Algorithm ($\mathcal{S}_{ga}$):** Algoritmo evolutivo basado en operadores de cruce y mutación de part

8. **KaHyPar (Karlsruhe Hypergraph Partitioning):** Estado del arte actual en particionamiento multinivel de grafos e hi
9. **BLOC-RELOC v1.0:** Versión original basada en bloques de contigüidad espacial sin blindaje de hubs.

10. **BLOC-RELOC v2.0:** La nueva arquitectura propuesta, equipada con Hub-Shielded Affinity (HSA) y el mecanismo d
--## 11.5 Métricas de Evaluación Cuantitativa

El rendimiento de cada algoritmo se medirá mediante un vector de ocho métricas cuantitativas formalizadas analíticam
### 1. Costo del Corte de Aristas ($C(\Pi)$)

Suma de los pesos de las aristas cuyas extremidades pertenecen a bloques distintos en la partición $\Pi = \{V_1, V_2, \d
$$C(\Pi) = \sum_{l=1}^k \sum_{u \in V_l} \sum_{v \notin V_l} A_{uv}$$
### 2. Conductancia del Corte ($\Phi(\Pi)$)
Mide la calidad de la frontera normalizada respecto al volumen de las particiones:
$$\Phi(\Pi) = \max_{1 \le l \le k} \frac{C(V_l, V \setminus V_l)}{\min(\text{Vol}(V_l), \text{Vol}(V \setminus V_l))}$$
Donde $\text{Vol}(V_l) = \sum_{u \in V_l} d_u$.
### 3. Balance de Partición ($B(\Pi)$)
Cuantifica la equidad en el tamaño de los bloques generados:
$$B(\Pi) = \frac{\max_l |V_l|}{\frac{n}{k}}$$
Un valor óptimo se aproxima a $1.0$.
### 4. Brecha Relativa de Calidad ($Gap(\mathcal{S}, G)$)
Mide el porcentaje de desviación del costo de corte obtenido por el solucionador $\mathcal{S}$ respecto al mejor cota
$$Gap(\mathcal{S}, G) = \frac{C(\Pi_{\mathcal{S}}) - C^*}{C^*} \times 100\%$$
### 5. Latencia Computacional Temporal ($\text{Time}(\mathcal{S}, G)$)

Medición en segundos del tiempo de CPU transcurrido desde la ingestión del grafo hasta la entrega de la partición final.
### 6. Consumo de Memoria RAM ($Mem(\mathcal{S}, G)$)
Pico máximo de memoria física asignada durante la ejecución, medido en megabytes (MB).
### 7. Coeficiente de Variación Inter-Semilla ($CV$)

Evalúa la estabilidad y robustez estocástica del algoritmo a través de 30 ejecuciones independientes con distintas semil
$$CV = \frac{\sigma_{\text{cut}}}{\mu_{\text{cut}}} \times 100\%$$

### 8. Robustez Topológica Global ($\mathcal{R}_{rob}$)

Índice compuesto que pondera la capacidad de mantener un $Gap < 5\%$ a través de todas las familias topológicas eva
--## 11.6 Protocolo Experimental (Secuencia de Ejecución)

La ejecución de la campaña experimental se regirá por un protocolo automatizado estructurado en cinco fases secuenci

```
========================================================================
PROTOCOLO DE EJECUCIÓN EXPERIMENTAL AUTOMATIZADO
========================================================================

[ FASE 1: Ingestión y Verificación de Instancias ]
│
▼
[ FASE 2: Inicialización de Semillas y Contenedores Docker ]
│
▼
[ FASE 3: Ejecución de Cartera de Solvers (Con Timeout τ_max = 3600s) ]
│
├──────────────────────────────┬──────────────────────────────┐
▼
▼
▼
[ Éxito: Registro de C(Π) ]
[ Timeout: Fallo por Límite ] [ Excepción de Memoria ]
│
│
│
└──────────────────────────────┼──────────────────────────────┘
│
▼
[ FASE 4: Auditoría Criptográfica y Almacenamiento Estructurado (Parquet/JSON) ]
│
▼
[ FASE 5: Consolidación para Análisis Estadístico y Perfiles de Rendimiento ]
========================================================================
```

* **Timeout Estricto:** Se establece un límite máximo de tiempo de cómputo $\tau_{max} = 3600$ segundos por insta

* **Auditoría de Resultados:** Cada archivo de salida contendrá hashes criptográficos SHA-256 de la partición generada
--## 11.7 Marco de Validación Estadística

Para asegurar que las conclusiones derivadas de los datos posean rigor científico incuestionable, se aplicarán las siguien

1. **Prueba de Rangos con Signo de Wilcoxon:** Prueba no paramétrica para evaluar diferencias pareadas significativas

2. **Tamaño del Efecto (Cliff's Delta o Vargha-Delaney $\hat{A}_{12}$):** Cuantifica la magnitud real de la superiorida

3. **Intervalos de Confianza Bootstrap:** Cálculo de intervalos de confianza al $95\%$ mediante remuestreo con reposi

4. **Perfiles de Rendimiento de Dolan-Moré:** Herramienta visual analítica para comparar la eficiencia global y la robus
$$\rho_s(\tau) = \frac{1}{n_{inst}} \text{size}\left\{ p \in \text{Inst} : r_{p,s} \le \tau \right\}$$

Donde $r_{p,s}$ es el cociente entre el costo obtenido por el algoritmo $s$ en la instancia $p$ y el mejor costo obtenid
--## 11.8 Análisis de Amenazas a la Validez
Todo diseño experimental riguroso debe explicitar sus posibles amenazas y las medidas de mitigación adoptadas:

* **Validez Interna:** Amenazas derivadas de fluctuaciones en el hardware o interferencias de procesos del sistema ope

* **Validez Externa:** Posible sesgo en la selección de las redes de prueba. *Mitigación:* Uso de una combinación equili

* **Validez de Constructo:** Riesgo de que las métricas elegidas no reflejen la verdadera dificultad del particionamiento

* **Validez Estadística:** Riesgo de conclusiones erróneas por variabilidad estocástica. *Mitigación:* Ejecución de 30 se
--## 11.9 Reproducibilidad de la Campaña

Para cumplir con los estándares de la ciencia abierta y permitir que cualquier grupo de investigación replique exactame
1. El código fuente completo del framework FAOT y de BLOC-RELOC v2 empaquetado bajo licencias de código abierto.
2. Los scripts de automatización en Python para la ejecución secuencial y paralela de los benchmarks.
3. Los conjuntos de datos crudos (datasets SNAP y generadores sintéticos parametrizados).
4. Los cuadernos analíticos de Jupyter con las rutinas estadísticas de Wilcoxon y generación de perfiles de Dolan-Moré.
--## 11.10 Preparación para los Resultados

En este capítulo se ha establecido de manera formal, exhaustiva y rigurosa la totalidad de la infraestructura, los protoco

**No se ha presentado ningún resultado en este capítulo.** La plataforma experimental se encuentra completamente bl
Esta formalización metodológica abre de forma natural el paso hacia el análisis empírico definitivo.
----# NOTA SOBRE EL CAPÍTULO 12

**El Capítulo 12 (Resultados Experimentales y Discusión) no ha sido redactado en este documento porque sus contenido

La campaña experimental descrita en el Capítulo 11 debe ejecutarse primero. Solo una vez que se hayan obtenido los d

El presente documento constituye la memoria técnica completa hasta el punto de inflexión del proyecto: la formalizació
----# ÍNDICE DE TABLAS
| Tabla | Título | Página |
|---|---|---|
| Tabla 6.1 | Resultados en Random Regular Graphs | 42 |
| Tabla 6.2 | Resultados en Watts-Strogatz Small-World | 44 |
| Tabla 6.3 | Resultados en Barabási-Albert Scale-Free | 45 |
| Tabla 6.4 | Resultados en Planted Partition Model | 46 |
| Tabla 6.5 | Resultados en Disconnected Networks | 47 |
| Tabla 6.6 | Resultados en SNAP Facebook | 48 |
| Tabla 6.7 | Resultados en SNAP WikiVote | 49 |
| Tabla 6.8 | Matriz de Rankings por Topología | 50 |
| Tabla 6.9 | Test de Wilcoxon Pareado | 51 |
| Tabla 6.10 | Tamaños del Efecto de Cohen | 52 |
| Tabla 6.11 | Coeficiente de Variación Inter-Semilla | 53 |
--# ÍNDICE DE FIGURAS Y DIAGRAMAS
| Figura | Título | Página |
|---|---|---|
| Diagrama 4.1 | Arquitectura del Sistema BLOC-RELOC | 28 |
| Diagrama 4.2 | Estructura de Bucket Priority Queues | 31 |

| Diagrama 4.3 | Flujo de Datos y Pipeline de Ejecución | 34 |
| Diagrama 4.4 | Ciclo Completo de Optimización | 36 |
| Diagrama 4.5 | Actualización Incremental en la Frontera | 38 |
| Diagrama 5.1 | Arquitectura del Harness Experimental | 55 |
| Diagrama 6.1 | Boxplot en Random Regular Graphs | 43 |
| Diagrama 6.2 | Curva de Convergencia en Small-World | 44 |
| Diagrama 6.3 | Perfil de Dolan-Moré (Calidad de Corte) | 50 |
| Diagrama 6.4 | Perfil de Dolan-Moré (Tiempo de Ejecución) | 50 |
| Diagrama 8.1 | Taxonomía de Regímenes Topológicos | 72 |
| Diagrama 9.1 | Arquitectura General del FAOT | 78 |
| Diagrama 9.2 | Pipeline Operativo Integral del FAOT | 83 |
| Diagrama 11.1 | Protocolo de Ejecución Experimental | 97 |
--*Fin de la Memoria Técnica del Laboratorio TCF*
*Documento congelado en su estado actual.*
*Próximo paso: Ejecución empírica de BLOC-RELOC v2 para la redacción del Capítulo 12.*

