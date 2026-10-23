# Guion para público general — versión divulgación

Objetivo: que cualquier persona salga entendiendo (1) el problema, (2) la idea del mapa,
(3) que el control de calidad funciona, y (4) que lo simple ganó.

**Regla de oro**: nunca digas *embedding, macro F1, n-shot, oversampling, LOF, p-value,
Bonferroni, baseline* sin traducirlo al instante. Las slides ya vienen traducidas; el
riesgo es que TÚ los digas por costumbre.

---

## 1. Introducción

**«¿Estamos escuchando todas las voces?»**
- Partir con UN ejemplo y quedarse ahí (recomendado: el mensaje de auxilio entre miles
  de consultas). No leer los 4 bullets.
- Mensaje: lo raro es lo que más importa, y es justo lo que casi no tenemos.

**«Por qué los clasificadores se inclinan…»**
- Analogía del alumno flojo: *«si el 95% de las preguntas de la prueba son de un solo
  tema, te aprendes ese tema y apruebas — sin haber aprendido nada del resto»*.
- El contraste 95% acierto / 0% detección es EL punto. Pausa dramática ahí.

**«¿Qué se hace para este desbalance?»**
- Analogía de la cancha despareja: o sacas jugadores del equipo grande, o le sumas al
  chico. Sacar es botar información; sumar exige inventar jugadores nuevos.
- Cierre: "esta tesis es sobre cómo inventar bien".

## 2. Estado del arte

**«La pieza clave: convertir cada texto en un punto de un mapa»** ⭐ SLIDE NUEVA
- ES LA SLIDE MÁS IMPORTANTE DE LA CHARLA. Si esta se entiende, todo lo demás fluye.
  Tómate 2 minutos aquí, no 30 segundos.
- Entra por la comida, que es lo que se ve en pantalla: *«Supongamos que quiero ordenar
  comidas. Le hago tres preguntas a cada una: ¿cuánto tiene de postre? ¿cuánto de
  sándwich? ¿cuánto de líquido? Con eso solo, cada comida ya tiene un lugar. Y miren:
  la pizza queda al lado del hot dog, porque se parecen. El borscht —una sopa— se va al
  otro extremo. Nadie le dijo al mapa que la pizza y el hot dog se parecen: salió solo
  de las respuestas.»*
- Deja que la gente mire el dibujo unos segundos: la comida engancha y da risa.
- Luego el salto: *«Con el texto es idéntico, con dos diferencias. Una: no son 3
  preguntas, son cientos. Dos: no las elegimos nosotros — el modelo las aprende solo,
  leyendo millones de textos. "Ganaron 2 a 0" y "victoria agónica en el clásico" caen
  casi en el mismo lugar, aunque no compartan ninguna palabra.»*
- Remate: *«Eso es todo lo que hay que recordar en esta charla: cerca = parecido.»*
- Imagen del Google ML Crash Course (CC BY 4.0), **original en inglés**. Traduce los
  tres ejes en voz alta al señalarlos, no los leas en inglés:
  *dessertness* = cuánto tiene de postre · *sandwichness* = cuánto de sándwich ·
  *liquidness* = cuánto de líquido. Los nombres de las comidas se entienden solos.
- A partir de aquí, usa siempre la palabra **mapa** (nunca "espacio de embeddings").

**«Camino 1 — SMOTE»**
- Analogía del color: *«si tengo pintura roja y pintura naranja, puedo crear los tonos
  intermedios. Nunca voy a inventar el azul.»* Eso es SMOTE: promedios, nada nuevo.
- No detallar k-vecinos ni interpolación. "Inventa puntos a mitad de camino entre dos
  ejemplos reales" basta.

**«Camino 2 — Pedirle los ejemplos a una IA»**
- Aquí conectas con lo que la audiencia ya conoce: *«esto es literalmente pedirle a
  ChatGPT: 'escríbeme 50 mensajes de spam como estos'»*.
- El punto ciego: *«la IA escribe bonito, pero nadie revisa si lo que escribió sirve.
  Bien escrito no es lo mismo que buen ejemplo.»*

**«El dilema»**
- Frase corta: *«uno tiene control sin novedad; el otro, novedad sin control»*.

**«La idea de esta tesis»**
- Antes de entrar, di el hueco en voz alta (ya no hay slide para esto): *«hoy todo lo
  que escribe la IA se acepta sin revisar, y todo pesa igual al entrenar»*.
- La idea en una frase: *«pedirle a la IA más de lo que necesito, ubicar cada texto en
  el mapa, y quedarme solo con los que caen cerca de los ejemplos reales»*.
- Analogía del casting/portero: *«la IA manda 150 candidatos, el mapa hace la
  entrevista, y solo 50 quedan contratados»*.

## 3. Hipótesis
- No leerla textual. Versión hablada: *«apostamos a que ese control de calidad mejora
  al clasificador más que el método clásico — y que no es suerte»*.

## 4. Metodología

**Pipeline** — recorrerlo con el dedo, 30 segundos. Es un diagrama de flujo, no más.

**«8.000+ experimentos»**
- El número es el mensaje: *«no lo probamos una vez: lo probamos ocho mil veces,
  cambiando todo lo que se podía cambiar»*.

**«Una nota justa: macro F1»**
- Analogía (chilena) que ya está en la slide: promedio de notas POR RAMO. *«Sacarse un
  7 en el ramo fácil no tapa el 2 en el difícil. Cada clase es un ramo, tenga 16
  alumnos o 720.»*

**«Cuatro maneras de hacer el control de calidad»**
- Son cuatro porteros: el que deja pasar a todos, el que mira si llegaste acompañado
  (densidad), el que mide qué tan cerca vives (cercanía), y el doble-check exigente.

**«La cascada: cuatro preguntas»**
- Leer las 4 preguntas tal cual (están en lenguaje natural). Rematar con el spoiler:
  *«guarden esta: la primera pregunta —la más tonta— va a resultar ser la única
  necesaria»*.

**«Ponderación suave»**
- Analogía del profesor que ya está en la slide. Alternativa: jurado de un concurso —
  *«todos votan, pero el voto del experto vale más que el del público»*.

## 5. Resultados

**«Resultado principal»**
- Decir la nota en simple: *«de un 71 a un 73 y medio, en promedio, sobre miles de
  experimentos. Y comprobamos que no es azar: la probabilidad de que sea suerte es
  menos de 1 en 10.000»*.

**t-SNE (2 slides)** — el "antes y después"
- *«Esto no es un diagrama: son datos reales dibujados en el mapa. Izquierda: todo lo
  que escribió la IA — miren cuántos caen en tierra de nadie. Derecha: lo que sobrevive
  al filtro.»*

**«Retornos decrecientes»**
- *«Con 10 ejemplos por clase, el método brilla. Con 50, ya casi no hace falta. Es el
  remedio para cuando hay poco — no un esteroide universal.»*

**«Por clasificador»**
- Mensaje corto: funciona con casi todos los tipos de modelo. No detenerse en SVC/Ridge.
- Si preguntan por Random Forest: *«hay una familia de modelos que ignora los pesos;
  su pariente moderno (XGBoost) sí los usa, y ahí la mejora es todavía mayor»*.

**«Número de clases»**
- Analogía: *«con 150 clases el mapa queda dividido en territorios tan chicos que
  'cerca' y 'lejos' dejan de significar algo»*.

**«Robustez»**
- *«Cambiamos la IA cinco veces y el mapa cuatro veces: da lo mismo. El mérito no es de
  ninguna IA en particular — es de la idea.»*

> **Slides ocultas en esta versión** (`\iffalse` en el .tex): «simple > complejo» y
> «generalización a NER». Si igual quieres mencionarlas de palabra:
> - *Simple > complejo:* «armamos filtros cada vez más sofisticados… y el más tonto
>   ganó. El portero perfecto deja la fiesta vacía.»
> - *NER:* «el mismo filtro, sin cambiarle una línea, funcionó en otra tarea distinta.»

## 6. Cierre
- Receta: dejarla proyectada y decirla en 20 segundos. Es la slide "llévate esto".
- Costo: *«centavos de dólar por dataset»* — siempre sorprende, úsalo.
- Cierre sugerido: *«La IA generativa es una fábrica infinita de ejemplos. Lo que
  faltaba era el control de calidad a la salida de la fábrica. Y resultó que basta
  una huincha de medir.»*

---

## Preguntas probables de público general

- **¿Esto sirve para español?** — Lo probamos en inglés; probarlo en español es trabajo
  futuro. La idea no depende del idioma: existen mapas para más de 100 idiomas.
- **¿La IA no inventa mentiras?** — Puede, y por eso existe el filtro: lo que cae lejos
  de los ejemplos reales se descarta, diga lo que diga.
- **¿Y si la IA se entrenó con esos mismos datos?** — (backup B8) El filtro no le
  pregunta nada a la IA: solo mide posiciones en el mapa, con otro modelo independiente.
- **¿Reemplaza a los datos reales?** — No: los datos reales siempre pesan más (peso
  máximo). Lo sintético es relleno de calidad controlada, no reemplazo.
- **¿Por qué no usar directamente ChatGPT para clasificar?** — Costo y escala: aquí la
  IA se usa unas decenas de veces por clase, una sola vez; el clasificador resultante
  es chico, rápido y gratis de operar.

## Cambios hechos respecto a la versión de defensa (2026-07-30)

1. Slide nueva «el mapa de significados» al inicio del estado del arte.
2. Eliminadas las fórmulas (SMOTE, macro F1, cascada, ponderación) → intuición + analogías.
3. Jerga traducida en todas las slides (recall→detección, win-rate→% de victorias, etc.).
4. Comentadas con `\iffalse` (recuperables): «Familia 2 — aumentación textual ligera» y
   «Familia 3.5 — fine-tuning».
5. Slides de backup intactas.
6. Slides desactivadas con `\iffalse` (recuperables): «Familia 2 — aumentación textual
   ligera», «Familia 3.5 — fine-tuning», «Tres deficiencias en los métodos actuales»,
   «Hallazgo contra-intuitivo: simple > complejo», «Generalización a NER».
7. En «Hipótesis y objetivos» quedaron solo H1 y el objetivo general.

La versión defendida quedó en git: commit `6283ed5`.
