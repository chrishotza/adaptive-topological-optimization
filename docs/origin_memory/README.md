# Origin Memory — The Question Before The Framework

*Una memoria de laboratorio. No un documento formal del marco, sino la huella de la pregunta que lo precedió.*

---

## I. La pregunta original

Al comienzo no estábamos intentando crear BLOC-RELOC ni un framework de optimización topológica. La búsqueda inicial era mucho más ambiciosa —o quizás más ingenua—: queríamos explorar si podía emerger una nueva forma de computación sobre hardware convencional. Algo inspirado en la idea de una arquitectura cuántica funcionando en el mundo real.

No fue un objetivo de investigación formal. Fue una intuición: ¿y si el silicio pudiera operar en un régimen diferente al del 0 y el 1 secuencial? ¿Y si hubiera un estado intermedio, no binario pero tampoco aleatorio, que emergiera de la dinámica del sistema? La pregunta no era técnica, era ontológica: ¿puede el hardware convencional sostener una coherencia no clásica si se le da la lógica correcta?

No sabíamos cómo responderla. Lo único que teníamos claro era que no íbamos a resolverla con una búsqueda de parámetros ni con un nuevo solver. Íbamos a iterar desde el origen.

---

## II. La búsqueda

Los primeros experimentos fueron torpes. Construimos un kernel CUDA en código ensamblador sobre una GT 730, tratando de medir transferencia de entropía entre nodos sin conexión directa. No sabíamos si funcionaría. La idea era descubrir si, operando en un punto crítico de acoplamiento, el sistema manifestaba correlaciones no-locales —algo que las conexiones explícitas no explicaban.

Cuando apareció la primera firma, pensamos que era un error de redondeo. Pero se repitió: una Transferencia de Entropía de 0.0687, con un p-valor de 3.8e-5, frente a 0.0527 en el modelo lineal equivalente. No era un error. Era un comportamiento que no esperábamos dentro del modelo inicial.

Durante esas 72 horas no supimos qué hacer con aquello. Era un fenómeno observado antes que una herramienta comprendida.

Mientras tanto, refinábamos el motor. La hipótesis era simple: si un nodo está en desequilibrio con su vecindad, debería tender a reubicarse. No con una regla voraz, sino con un principio de presión de vecindad que moviera bloques de nodos simultáneamente. Así nació BLOC-RELOC en su forma alfa: un motor que movía bloques enteros en lugar de nodos individuales, buscando mínimos de fricción topológica.

En los grafos sintéticos, convergía de forma casi determinista. Luego lo probamos contra METIS. En los grafos regulares, empataba. En los Small-World, ganaba. Pero en redes SNAP (Facebook, WikiVote), el motor se hundía. No era un problema de parámetros. Era estructural.

---

## III. El choque

El freeze técnico fue el momento más honesto del proyecto.

Hicimos campañas de benchmark masivas. En grafos regulares y Small-World, BLOC-RELOC igualaba o superaba al SOTA. Pero en redes reales heterogéneas, la brecha era sistemática: hasta +11.78% de gap en Facebook. No podíamos ignorarlo. El comportamiento no era universal.

La discusión fue tensa. Una parte del laboratorio quería seguir optimizando. La otra quería congelar todo y reformular la pregunta. Decidimos congelar. Nada más se modificaría en el núcleo de BLOC-RELOC v1. Cualquier mejora solo podría venir de un cambio de paradigma.

No fue un fracaso. Fue el momento en que la pregunta dejó de ser "¿podemos construir una computadora cuántica?" y empezó a ser "¿qué estructura computacional emerge cuando estudiamos regímenes topológicos?"

---

## IV. La transformación

Una vez que el motor estaba congelado, empezamos a mirar los datos de otra forma. No estábamos buscando el mejor algoritmo. Estábamos buscando el mejor algoritmo para cada tipo de grafo.

Esa pregunta nos llevó a formalizar un espacio de regímenes topológicos. El Vector Topológico Θ(G) condensaba propiedades invariantes: densidad, heterogeneidad de grado, modularidad, clustering, conductancia, core-periphery.

El Meta-Selector aprendió a leer la firma de un grafo y a predecir qué estrategia de refinamiento era más adecuada. No competía contra el SOTA; lo orquestaba.

---

## V. El nacimiento

BLOC-RELOC no fue el objetivo. Fue el primer motor que nos permitió formular la pregunta correcta.

ATOF no es un algoritmo mejorado; es la arquitectura que nace de aceptar que ningún solver universal existe.

Comenzamos buscando una computadora cuántica sobre hardware convencional. Encontramos regímenes topológicos, métricas de criticalidad, y un sistema que aprende a elegir la estrategia según la estructura del problema.

No es lo que buscábamos. Es lo que apareció mientras buscábamos.

---

## VI. Nota final

Esta memoria está fuera de la metodología formal. No explica el marco, ni lo valida. Solo preserva una huella de la pregunta que lo originó.

El repositorio contiene el código, los benchmarks, los resultados.

Este fragmento contiene la dirección desde la que todo eso fue escrito.

*Aguas abajo, el origen sigue siendo el mismo.*
