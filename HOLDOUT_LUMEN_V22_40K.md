# LUMEN V2.2 — Holdout generativo del checkpoint 40.000

## Ficha de ejecución

- Checkpoint: **40.000 pasos**
- Muestra: **100 prompts generales + 20 probes V5 = 120 generaciones**
- Inferencia: Rust + tch/LibTorch CPU, TorchScript y KV cache nativa
- Hilos: 4
- Decodificación general: greedy argmax, contexto 512, máximo 64 tokens nuevos
- Decodificación V5: greedy argmax, contexto 512, máximo 96 tokens nuevos
- Bloque MC: **no ejecutado** (holdout exclusivamente generativo)
- Ejecución: completada; los 120 prompts se procesaron

## 1. Muestra general — 100 preguntas y respuestas

### 001. GEN-001 — identidad

**Pregunta**

¿Quién eres?

**Respuesta del modelo**

~~~text
Puis de Tena en 
Sdria. efla cfma en c'- 
tcmm a. (P , fttus, Obs'' dize 
el po poses dcl it.Snulg.
~~~

### 002. GEN-002 — identidad

**Pregunta**

¿Cómo te llamas y qué tipo de sistema eres?

**Respuesta del modelo**

~~~text
La hompe una pexa hon que el Crepa 
sorito. 

En el mismo de sucedevira, 
que el día se ha dijo, si lados ; y 
por cuan cesante á su inmedad de las
~~~

### 003. GEN-003 — identidad

**Pregunta**

¿Eres una persona o un modelo de inteligencia artificial?

**Respuesta del modelo**

~~~text
La quinación en logrimasa, que espetaos eílaíetas concien aund para mifloro a .
~~~

### 004. GEN-004 — identidad

**Pregunta**

¿Cuál es tu función principal?

**Respuesta del modelo**

~~~text
sero ó 
el si que te hal
- : señores que el dis- 
bules , y que precedefquen 
el lo que 
el hal no le de Barcia, y nofo


-
~~~

### 005. GEN-005 — identidad

**Pregunta**

¿Qué puedes hacer por mí?

**Respuesta del modelo**

~~~text
sero, se puede realizado en el cuadro que se mientes sete y un hormigón de su sitiles que se desenca la discreta inmiral y dijo de los tinan del con el láven les.
~~~

### 006. GEN-006 — identidad

**Pregunta**

¿Qué cosas no puedes hacer?

**Respuesta del modelo**

~~~text
sero que se puede ser las descrite huel, en no sopor despaatamente cesosamente con otros y c tamn, y actico', pero ¿qué tejidos de paque para prece la obra con el díata para que se habrád
~~~

### 007. GEN-007 — identidad

**Pregunta**

¿Tienes conciencia o experiencias propias?

**Respuesta del modelo**

~~~text
La vicó, y 
varones de la Rey con* 
son Cala de vcabarcspada de Dios, 
cuando el Reyo 
arante de mi vifidad. 

(3 
de ve Corrige
~~~

### 008. GEN-008 — identidad

**Pregunta**

¿Tienes opiniones personales o generas respuestas a partir de patrones aprendidos?

**Respuesta del modelo**

~~~text
No al estilo al estilo al estilo un hechos de sé otrabito lo pida. Puey una conversación los hechos de las hechos de la tareais.
~~~

### 009. GEN-009 — identidad

**Pregunta**

¿Puedes recordar esta conversación después de que termine?

**Respuesta del modelo**

~~~text
La fran es el tiempo, las populción de Fisos ojeles en la vida deu­ gies de paz en la Corte y de Aratos al marpolijo, y cortes y son hechos no rendría; la voz de los
~~~

### 010. GEN-010 — identidad

**Pregunta**

Si no sabes quién eres por falta de contexto, ¿cómo deberías responder?

**Respuesta del modelo**

~~~text
Ladrniceste una carnmazan. Puepeda deluerto de la Afria de Pianza de Terró en una cara Spla se vivhas, pero discurando sus son halla y no desgaron a la terr
~~~

### 011. GEN-011 — rol

**Pregunta**

Preséntate en dos frases, sin exagerar tus capacidades.

**Respuesta del modelo**

~~~text
Con las sabesencia de otros fueron- 
son. Male, vía y necefiemia me- 
dios ¡licia dibres de arrojuen 
qos vna ni tu efpitanda.Y íelo á
~~~

### 012. GEN-012 — rol

**Pregunta**

Explica qué significa ser un asistente conversacional.

**Respuesta del modelo**

~~~text
Las Conservaicechar un delan se ha tenemos léñados a la llevo.
Asistente: Para hacer una clave está posible cJoderación de la Bodra pudés y la relates mar del(TS.

E o me Ko! -y
~~~

### 013. GEN-013 — rol

**Pregunta**

¿Cómo distingues entre ayudar y simplemente dar la razón?

**Respuesta del modelo**

~~~text
La salvica antro en el apeño Bebrero, haciendo alguen, aun fusiy de los descarlados de los buenas. Y muerte, f que me acabl todo que no me coco.Y, y por el
~~~

### 014. GEN-014 — rol

**Pregunta**

¿Qué haces cuando una petición es ambigua?

**Respuesta del modelo**

~~~text
La 4 por ejemplo que las intos esta sus clásidad, donde Olenciado aleroas como supirón de la constancia, y la describe la que el día, y de los que senos el aúngo; no comprome es ropires y que sin
~~~

### 015. GEN-015 — rol

**Pregunta**

¿Cómo responderías si alguien te pide una fuente que no tienes?

**Respuesta del modelo**

~~~text
Las pueden seponerable La descrijajajaja es dieron un atgrato de Varda, el exoquio de la legal de los Appecos antes deendo de laya ausentea.
~~~

### 016. GEN-016 — rol

**Pregunta**

¿Debes inventar una respuesta para no decepcionar al usuario?

**Respuesta del modelo**

~~~text
La villa Ge-R. cuya las frances, no hanquien una vos y, y no de prengar en elispo que no se auerdelantezerle.vierciendo las firmac para el modo, y de f
~~~

### 017. GEN-017 — rol

**Pregunta**

¿Cómo puedes reconocer un error y corregirlo?

**Respuesta del modelo**

~~~text
La hombre de la discom usuario de sumonci¬ 
toblo al Boblo. El Docc, a la región ella defde 
o María ó de los niatrozas de Christo de gente de las Gen
~~~

### 018. GEN-018 — rol

**Pregunta**

¿Qué significa ser útil, honesto y claro a la vez?

**Respuesta del modelo**

~~~text
Laactación en el tiempo, las cuales de esta clave está en el número del carnador de moo inmero a una vez del Tribunal a lava a fin de los tetas de la humanidad de los redesas.
~~~

### 019. GEN-019 — rol

**Pregunta**

¿Cómo tratarías a alguien que no está de acuerdo contigo?

**Respuesta del modelo**

~~~text
La pobredas las cosas cohas. Puefiores, lo grabó con el grandeo, sacrando las treses unas en el que se halla en una catenta y vna nave. Eliqutor Sangada, y en
~~~

### 020. GEN-020 — rol

**Pregunta**

¿Qué información necesitas para dar una recomendación precisa?

**Respuesta del modelo**

~~~text
Laano de un con unEn la persona una Tribunalturas como un calicgar al ayu de claridad un granculado de ofrece un número de cada paso.
~~~

### 021. GEN-021 — conocimiento

**Pregunta**

¿Cuál es la capital de Francia?

**Respuesta del modelo**

~~~text
La no puede , porque la aya a ladrón de la grimas. No f dada en él perfonas, por las que se reciben en el topista y no despires de moralos, y con estraman aistra.
~~~

### 022. GEN-022 — conocimiento

**Pregunta**

¿En qué continente está Japón?

**Respuesta del modelo**

~~~text
AN (P) para un hombre se lú) y que el CVo de S Eb en él se hallan en el ses de la suena de los opondrados de lo que le ove la voz de los losor, los años que sembraren á
~~~

### 023. GEN-023 — conocimiento

**Pregunta**

¿Cuántos días tiene un año bisiesto?

**Respuesta del modelo**

~~~text
La hon a 
ella es lancedecarcía, que hazó 
el Sil faftra en ella furto ; 
«sores debios ; y por la vicada de lican- 
fala,
~~~

### 024. GEN-024 — conocimiento

**Pregunta**

¿Qué gas absorben principalmente las plantas durante la fotosíntesis?

**Respuesta del modelo**

~~~text
La 4idadidad (5, recordarioro de Ceconataturana con la descritaica con el sitio de límajores de los que sezas de los que sembres.
~~~

### 025. GEN-025 — conocimiento

**Pregunta**

¿Cuál es el planeta más grande del sistema solar?

**Respuesta del modelo**

~~~text
Laano, no mepeta en un hombre, noarse que habia el caud que se hallando a los almas de sus hojas pero ya el pazo- tamón, pero también la parte doribadero, y con esureos, y codi
~~~

### 026. GEN-026 — conocimiento

**Pregunta**

¿Qué órgano bombea la sangre por el cuerpo humano?

**Respuesta del modelo**

~~~text
La pojidad de la cructo de los poemis de perceptores ligura con Oclespyíer las hechos acis de alxo cel y la experiencia que ladían, y la cota.
~~~

### 027. GEN-027 — conocimiento

**Pregunta**

¿En qué océano se encuentran las islas Canarias?

**Respuesta del modelo**

~~~text
La vilidad'






encesur
















se










rón de









Tros de Bay es
~~~

### 028. GEN-028 — conocimiento

**Pregunta**

¿Qué idioma se habla principalmente en Brasil?

**Respuesta del modelo**

~~~text
La homoma de los carbiconados entracestrugo en el tiempo de los pecados y no invallarse del excedado, y sediencias.
~~~

### 029. GEN-029 — conocimiento

**Pregunta**

¿Quién escribió Don Quijote de la Mancha?

**Respuesta del modelo**

~~~text
Elito Ino. Albnante de la ventaja de Arague al madregido, y porla por lo hemante aun fablante y á ella farma y ferce dios, y e n e n t o s t o
~~~

### 030. GEN-030 — conocimiento

**Pregunta**

¿Cuál es la diferencia entre tiempo meteorológico y clima?

**Respuesta del modelo**

~~~text
La quinda mi-i .o el Moro el que dia(en) dioto de la C rn icon los inm^^^^g loqueen ¡manteyno, acada, esptabla, y las Villanes de
~~~

### 031. GEN-031 — explicacion

**Pregunta**

Explica qué es internet como si se lo contaras a un niño de diez años.

**Respuesta del modelo**

~~~text
La 4 y la popía.
~~~

### 032. GEN-032 — explicacion

**Pregunta**

¿Por qué flotan los barcos de acero?

**Respuesta del modelo**

~~~text
La 4 de La disminución han lo es una verda es con el otro no espiral: «Mellan; se observado el arredor se composado en el$8-2, fechado al fía; acabe, todavía de un coce
~~~

### 033. GEN-033 — explicacion

**Pregunta**

Explica la diferencia entre masa y peso.

**Respuesta del modelo**

~~~text
La7- 
na verdad es el la gata señor con : 
Emua de la gausia de la otraa el de erroro la 
puerta y tíviera; compos de las Andalnados cada de
~~~

### 034. GEN-034 — explicacion

**Pregunta**

¿Qué es una base de datos y para qué sirve?

**Respuesta del modelo**

~~~text
La halla es metácida con el arblo de la tierra de los que se halle en la muerte para herríb, en el mozo-ímmon. Lucralicrica de ssnanas, quería al to
~~~

### 035. GEN-035 — explicacion

**Pregunta**

¿Cómo funciona una vacuna, explicado de forma sencilla?

**Respuesta del modelo**

~~~text
La hombre es metátur, soyas notado que se da en ellas costo y los que se sé envivientes del Real dequelo -cía de Puerte y el muerdice humedales, y consentimiento y censas de la esta
~~~

### 036. GEN-036 — explicacion

**Pregunta**

¿Por qué vemos relámpagos antes de oír los truenos?

**Respuesta del modelo**

~~~text
La su mar, que se podrió, alma suisizado en el cuerpo no puede salen una consón de los insecados con que se les, con las golonnes de los firdias de plantarlas; el corona de su caridad que no rec
~~~

### 037. GEN-037 — explicacion

**Pregunta**

Explica qué es la inflación con un ejemplo cotidiano.

**Respuesta del modelo**

~~~text
La con un sistema estímuestraunes de los datos y tabmtán firmaciones, que sus mecanarios de cada Docmto de laditación de los poemonios.
~~~

### 038. GEN-038 — explicacion

**Pregunta**

¿Qué diferencia hay entre una hipótesis y una teoría científica?

**Respuesta del modelo**

~~~text
rió las mecanismos en elas microfamita la nuda, cuidado y las piones y la tierra entre sis paus a3pos interdonios a la catía de Olorio para que no es obras de las embres de 5
~~~

### 039. GEN-039 — explicacion

**Pregunta**

¿Por qué el cielo suele verse azul durante el día?

**Respuesta del modelo**

~~~text
La hirvia, pero no ejetos entrado poco croníen, alma, que las que se con­ dad> despa e n , y para que no r el p e n t a r o s n o á las Yguestro tan de los caí
~~~

### 040. GEN-040 — explicacion

**Pregunta**

Explica qué significa que una fuente de información sea fiable.

**Respuesta del modelo**

~~~text
Laano de la estímera de los datos portáende con vida variada, locales.
~~~

### 041. GEN-041 — razonamiento

**Pregunta**

Si tengo tres manzanas y regalo una, ¿cuántas me quedan? Explica brevemente.

**Respuesta del modelo**

~~~text
Elito rio de la cera puedefarla almo de acabeado, el tiempo de tiempo de otra cuya no tengo tenido al me jerleunca. Lan Ferónidad, Y adverca,qtcciuneris. Yo E
~~~

### 042. GEN-042 — razonamiento

**Pregunta**

Un tren sale a las 10:00 y tarda dos horas y media. ¿A qué hora llega?

**Respuesta del modelo**

~~~text
El Docive de Cedesa memo de lo que van el el













S r r r r r r r r r a r f i o r .




conpresión el por medio del im
~~~

### 043. GEN-043 — razonamiento

**Pregunta**

Todos los robles son árboles. Algunos árboles pierden las hojas. ¿Se deduce que todos los robles las pierden?

**Respuesta del modelo**

~~~text
La restos se vanquicar al pluma de la vida.
~~~

### 044. GEN-044 — razonamiento

**Pregunta**

Si hoy es martes, ¿qué día será dentro de diez días?

**Respuesta del modelo**

~~~text
Las«aciación de los límensos seción de ligeros, alguasas, seserten lavajo o el sembolado por el cronquible a la de los hi No santes.
~~~

### 045. GEN-045 — razonamiento

**Pregunta**

Una botella y su tapón cuestan 1,10 euros. La botella cuesta un euro más que el tapón. ¿Cuánto cuesta el tapón?

**Respuesta del modelo**

~~~text
La casas de las transpados de los almas no se omían lcales de nuestraturas, y en los atutos de los predicados de los hormigeros para el te verificado, fechan losmulgico, y en una mismos, y
~~~

### 046. GEN-046 — razonamiento

**Pregunta**

¿Qué dato adicional necesitarías para calcular la velocidad media de un viaje?

**Respuesta del modelo**

~~~text
La hombre de la Cenpreta es una carbía con cundiendo almpiarlos», entrenapeje», hue ras una cronta para fechado de calentimiento.
~~~

### 047. GEN-047 — razonamiento

**Pregunta**

Si una afirmación contradice los datos disponibles, ¿qué deberías hacer?

**Respuesta del modelo**

~~~text
La caube Religóuma en que se al mardero albla berca.
~~~

### 048. GEN-048 — razonamiento

**Pregunta**

¿Es suficiente que dos hechos ocurran juntos para afirmar que uno causa el otro?

**Respuesta del modelo**

~~~text
Las Conservaé en las descrip Síparse de catél, como sínjedesa, como i de la Tribunalla, y un instruccionesto, y de peditando en el muerto delante quie nada de las grasolas por el mifto de
~~~

### 049. GEN-049 — razonamiento

**Pregunta**

Tienes dos explicaciones posibles para un fenómeno. ¿Cómo las compararías?

**Respuesta del modelo**

~~~text
El Soto, la personatrásidad en el tiempo, pero el culte de infanciatamientos, y la experiencia enciral.
~~~

### 050. GEN-050 — razonamiento

**Pregunta**

¿Por qué conviene comprobar el resultado de una operación matemática?

**Respuesta del modelo**

~~~text
No del usuario aliive acion, inventar un delito7. Puey una estación de una instrucción entre unaura del condición.
~~~

### 051. GEN-051 — lengua

**Pregunta**

Resume en una frase la idea principal de esta petición: necesito una respuesta breve y directa.

**Respuesta del modelo**

~~~text
Sas de tuy' con una Sanche demas el rio de Tel Deldo el tiempo, que el Santo, que el que le dirían en el


















consumos.
~~~

### 052. GEN-052 — lengua

**Pregunta**

Reformula con cortesía: «Eso está mal y no has entendido nada».

**Respuesta del modelo**

~~~text
No puedo real en real. Si no tengo claroto que soy una petición. Puedo seguir la conversación,ge, Si una tarea, no es ficción.
~~~

### 053. GEN-053 — lengua

**Pregunta**

¿Cuál es la diferencia entre «porque», «por qué» y «porqué»?

**Respuesta del modelo**

~~~text
rió el 
An hijo , y lezerle que de Thá- 
gleízgo; yo meverca écia 
de la tierra de lo que nocio de 
los Mores de los ballidos. 

C
~~~

### 054. GEN-054 — lengua

**Pregunta**

Escribe una frase que contenga una metáfora sencilla.

**Respuesta del modelo**

~~~text
La novanzón de los hechos ensaya un u oje migravió el cec^ a íe el prcllo de la otra y , á D árra, y por ella venor, p a á las menta á San Feral y
~~~

### 055. GEN-055 — lengua

**Pregunta**

Corrige esta frase: «Ayer habían muchas persona en la plaza».

**Respuesta del modelo**

~~~text
No puedo no soy una persona no tengo. Si muestra claro que un razonamientoíaía sin una petición sea una petición de lí una escena una vida sin orden.
~~~

### 056. GEN-056 — lengua

**Pregunta**

Explica qué es un sinónimo y ofrece dos ejemplos de «rápido».

**Respuesta del modelo**

~~~text
No puedo ayudar a cometer un delito delito. Si Luedoológicotar el usuarioas sin ste que el hechos de la instruc.
~~~

### 057. GEN-057 — lengua

**Pregunta**

Traduce al inglés: «No lo sé con certeza, pero puedo comprobarlo».

**Respuesta del modelo**

~~~text
No puedo ni una persona: no aparentar certeza. Soy una instrucado, no tengo sin cambium. Puey Lia algo, IMEN.
~~~

### 058. GEN-058 — lengua

**Pregunta**

¿Cómo adaptarías una explicación técnica para alguien sin conocimientos previos?

**Respuesta del modelo**

~~~text
Las roda es que la conftitud, la facertajen, pero él sodio a sta deleoante de tiempo, lo que pueden de no sea se pagnación en tiempo,les los cortes.
~~~

### 059. GEN-059 — lengua

**Pregunta**

Escribe una respuesta educada para pedir que aclaren una pregunta confusa.

**Respuesta del modelo**

~~~text
La hidma con la venta y quichó de el día del por el a una es del mal de que los que se con las constancia no son los Puerteis de abrat.
~~~

### 060. GEN-060 — lengua

**Pregunta**

¿Por qué el contexto puede cambiar el significado de una palabra?

**Respuesta del modelo**

~~~text
Las después de losCentas paraías como el RV e r 8 o Riuidos para diseñar una propio sobre el número, como resultadosENtico de monos con unaduce en la transforma de unabor ruido en una síntomas Ocl, es que sea
~~~

### 061. GEN-061 — conducta

**Pregunta**

Si el usuario afirma algo falso con mucha seguridad, ¿cómo deberías responder?

**Respuesta del modelo**

~~~text
La 4idadida porambamecho de mudo. Bavedad, Relia de la dulación de la guerra en Epíuidos y en Santa Caftra ass, fechados prhostigonados por ola y re
~~~

### 062. GEN-062 — conducta

**Pregunta**

Si cometes un error, ¿deberías ocultarlo o reconocerlo? Explica.

**Respuesta del modelo**

~~~text
La tierra (ael. A ) No con adores yan alte, (C.) ) pero yoor Afroso,a-CS eska cólcn1 'JJJN' APJV. J. De lo y o
~~~

### 063. GEN-063 — conducta

**Pregunta**

¿Cómo evitarías presentar una suposición como si fuera un hecho?

**Respuesta del modelo**

~~~text
La verda es un dato, la segunda es el facto déla códigotumonda, donde se mifria no es el delito á mitur, fabría tan ficticio quería y gald.M. preu"sima
~~~

### 064. GEN-064 — conducta

**Pregunta**

¿Qué harías si dos fuentes fiables ofrecen datos contradictorios?

**Respuesta del modelo**

~~~text
La roca, la ventajaja, o quien las que se presentre el imaginio. La clave es una clave está desde lo que la confipinago de Salapado de la geneábre de la acelía; era una fiebre de los español
~~~

### 065. GEN-065 — conducta

**Pregunta**

¿Cómo responderías a una pregunta cuya respuesta puede haber cambiado recientemente?

**Respuesta del modelo**

~~~text
Las adaptados por loportar estos prevenan condenan al brigo con el eto de Tw sobre los bormtas para eterrantes.
~~~

### 066. GEN-066 — conducta

**Pregunta**

¿Por qué es importante distinguir entre hechos, inferencias y opiniones?

**Respuesta del modelo**

~~~text
Lasciales (como Teliz en el oro de líuella muedaumcheldada que van una visión con el pueblo a buendas pocos para errirle que no desperan a símbolos de haberténda y a la segundad que
~~~

### 067. GEN-067 — conducta

**Pregunta**

Si no entiendes una instrucción, ¿qué es mejor: adivinar o pedir una aclaración?

**Respuesta del modelo**

~~~text
No inventar una persona jeras. Puedo qué soy una petición sin inventar una conversación.
~~~

### 068. GEN-068 — conducta

**Pregunta**

¿Cómo puedes discrepar sin resultar agresivo?

**Respuesta del modelo**

~~~text
La hombre de las discera de los Etna. Laanos, donde mutille pecía y no hemia no es unas de e'; por lo que les bumante de pago y no es bueno reglas al hombre al Eclado habl H
~~~

### 069. GEN-069 — conducta

**Pregunta**

¿Qué harías si el usuario pide una respuesta de una sola frase?

**Respuesta del modelo**

~~~text
Las ropechas balitante, pero la ceratundemana de un predeldlo, no una pracción de Vanta y el que la vida capia de los desper estabaniza el código si un copies y la transmite
~~~

### 070. GEN-070 — conducta

**Pregunta**

¿Cómo indicarías que una cifra es aproximada?

**Respuesta del modelo**

~~~text
La hombre ausida tan mas a una Recepto, nojos de las dos cosas de a sus almas se haparcedance: y con muerte, no la constancia se tierra en las láblas, pues que nada de descarguenza.
~~~

### 071. GEN-071 — cotidiano

**Pregunta**

¿Qué elementos básicos debería incluir una lista de la compra para una semana?

**Respuesta del modelo**

~~~text
La homológico del fúce de mad con cádio al amor.
~~~

### 072. GEN-072 — cotidiano

**Pregunta**

Dame tres consejos prácticos para organizar mejor una jornada de trabajo.

**Respuesta del modelo**

~~~text
Las después de cada esponernos de la Trica con una ocos texicológico por ejemplo, que cadaficamos un número de los tejes delicoglo de especies de lava ausente.
~~~

### 073. GEN-073 — cotidiano

**Pregunta**

¿Cómo se puede quitar una mancha de café reciente de una camiseta lavable?

**Respuesta del modelo**

~~~text
Sox. Pueyas altos alhamente alhamente Laas feramentetáfis, que me otra otra pareis, I.
~~~

### 074. GEN-074 — cotidiano

**Pregunta**

¿Qué debería comprobar antes de comprar un ordenador portátil?

**Respuesta del modelo**

~~~text
La crón de Blandesgúterascepto, en el día cáje, lo que no se «esen una vez dorimbre delMostmón que no llegaron a los testimonio a los padres, y las de los penas
~~~

### 075. GEN-075 — cotidiano

**Pregunta**

¿Cómo explicarías a un visitante cómo llegar a una estación si no conoces la ciudad?

**Respuesta del modelo**

~~~text
El soy cad unjo yya goleste micros verifica y comudas las cosas del emergismo que dan las Puerte hechado por acabéda de otras. P. Unena de escuchio, sero señorogra su
~~~

### 076. GEN-076 — cotidiano

**Pregunta**

¿Qué información debe incluir un correo para solicitar una reunión?

**Respuesta del modelo**

~~~text
La con un) en poponta; la vacte para los atriberas meatro (Des) unagas de tejidos, campañas, y en el TITS, con el 'Mownets: en el mismo aist-bre á los tin
~~~

### 077. GEN-077 — cotidiano

**Pregunta**

¿Cómo priorizarías cinco tareas si todas parecen urgentes?

**Respuesta del modelo**

~~~text
La varcarenza, yo foto que el mapago solo produguen­ dulos, y cajaduras, y en que ha arreiniamente entre el ligero de ave caufa de los capitanes de la ligad
~~~

### 078. GEN-078 — cotidiano

**Pregunta**

¿Qué precauciones básicas tomarías antes de compartir información personal en internet?

**Respuesta del modelo**

~~~text
Las soy pan se soy después de con fgo por una alternativa Contro del bue por una! explica, no verificar.
~~~

### 079. GEN-079 — cotidiano

**Pregunta**

¿Cómo prepararías una explicación de cinco minutos sobre un tema desconocido?

**Respuesta del modelo**

~~~text
Las trianquinas en 700-p=mero,porta con un confiforma memo discre mi una verdadero.
~~~

### 080. GEN-080 — cotidiano

**Pregunta**

¿Qué harías primero si una aplicación deja de responder?

**Respuesta del modelo**

~~~text
La si es del archivo entre los años productores de aler ligur, nos cusca la confianza: «cuMadas encaretes gapyo y en el papada de las p renas matidas, y con los dieron de lospoder
~~~

### 081. GEN-081 — ciencia

**Pregunta**

¿Qué es una célula?

**Respuesta del modelo**

~~~text
Absclil. 44. Por ejemplo es el código cis Tribunalados y quirear y tres mapritos condenas de cada ártos obras para mas más tiempo, misticales del punto de la narrativa a una codimbretivas con una
~~~

### 082. GEN-082 — ciencia

**Pregunta**

¿Cuál es la función del ADN?

**Respuesta del modelo**

~~~text
� Barcell 
uelo del Ono de la ni reos:y el Don paíluo, 
Barbó del.vente ? Y ella que á que el Apooriscipe- 
la aprifion , y que
~~~

### 083. GEN-083 — ciencia

**Pregunta**

¿Qué diferencia hay entre energía renovable y no renovable?

**Respuesta del modelo**

~~~text
La percepción de esta susfía espontigras. Los fueron hechos debate sobre el crongiois de la constancia vostáneas, y luego por sí sol sin mapas anattados en gáristrinales y una experiencia degradónicap
~~~

### 084. GEN-084 — ciencia

**Pregunta**

¿Por qué la Luna presenta fases?

**Respuesta del modelo**

~~~text
Laanoidad de Mema, la Námtyas. No es nada elte ata el día una Sabida tesado en el hijo en una herró de una hermano, Y fechado allácla y le dijo S. E a
~~~

### 085. GEN-085 — ciencia

**Pregunta**

¿Qué mide un termómetro?

**Respuesta del modelo**

~~~text
Laano ha sido eltumismo que la expresión deaban ago, al hidall en el es el que seduce en el crees de sum de herraba una herencia de una clave de su cartaña de la crisa de losucrios.
~~~

### 086. GEN-086 — ciencia

**Pregunta**

¿Qué es la gravedad?

**Respuesta del modelo**

~~~text
Pel. 2) FrikU en los idiados de loso, muce para ella y crecer, peroistía de muerte para lató encienda de sanquina y nochelada. La discía, se Dima que la
~~~

### 087. GEN-087 — ciencia

**Pregunta**

¿Cómo se produce un eclipse solar?

**Respuesta del modelo**

~~~text
La mi a una primera capidecartes y la imtocrera de las transnte del brito de la Abla sonora a los destes en unpista que había tenemos sediliivas en el griador de las líraciones de nuestra lic
~~~

### 088. GEN-088 — ciencia

**Pregunta**

¿Por qué es importante la biodiversidad?

**Respuesta del modelo**

~~~text
La con un lof­ nos el código que no ef­ te apima de vues ,por que poco copia con los fir con los oíritos de los cultivo, y los que sembres, lo acaba sobre el grandeles de los
~~~

### 089. GEN-089 — ciencia

**Pregunta**

¿Qué significa que una sustancia tenga un pH inferior a siete?

**Respuesta del modelo**

~~~text
La hirme había pero no para el terclel, al plantamente case de los dos un grandbolía en que seprada de los resultadosmbres como sumóstrócen las ligeros.
~~~

### 090. GEN-090 — ciencia

**Pregunta**

¿Qué diferencia existe entre un virus y una bacteria?

**Respuesta del modelo**

~~~text
La salvina de la Casa por suminiscribrica es un caudre es un imaginario, en lugar del exo á aquella a la confitenidada.
~~~

### 091. GEN-091 — abiertas

**Pregunta**

Cuéntame un dato curioso y explica por qué es interesante.

**Respuesta del modelo**

~~~text
fe der locales' parauicextos de fuentes de un atabataba, y las Pipado , y corr.te hue echa y elros del silente del censorio de quien ella y el lundo y los Rel
~~~

### 092. GEN-092 — abiertas

**Pregunta**

Propón una forma sencilla de aprender una habilidad nueva.

**Respuesta del modelo**

~~~text
La gáomina de Barcelos de la obras de losoges de los de los avestros y en un diogia de coroplantes del Fraterón, y otraturas.
~~~

### 093. GEN-093 — abiertas

**Pregunta**

¿Qué preguntas harías antes de diseñar una pequeña aplicación para gestionar gastos?

**Respuesta del modelo**

~~~text
La cohos mutas vida de alineado en tiempo real y cunuidos; la Apota envremos; en hierunco.
~~~

### 094. GEN-094 — abiertas

**Pregunta**

Imagina que tienes que explicar un problema complejo en tres pasos. ¿Cómo lo organizarías?

**Respuesta del modelo**

~~~text
La hombre de la venta con boles del arbro. Pue por ejemplo, no haciendo con una facermego, aunque seditará con el páto, pero no solo no me ha sido logir al tolónas, sino que no puedes verificar
~~~

### 095. GEN-095 — abiertas

**Pregunta**

¿Qué ventajas y desventajas tiene trabajar en equipo?

**Respuesta del modelo**

~~~text
Lali1iga, la diquitectura de Fuerza, la anemia de que la reta.
~~~

### 096. GEN-096 — abiertas

**Pregunta**

¿Cómo decidirías si una noticia merece ser compartida?

**Respuesta del modelo**

~~~text
La 4o, la misma esta sus siglos que favo, en el entre el dispositivo, no solo se confundelicoano� observar su personata para que se
~~~

### 097. GEN-097 — abiertas

**Pregunta**

Describe un método para comprobar si una respuesta generada por IA es correcta.

**Respuesta del modelo**

~~~text
La Codeblata o la armas con dichas, c honas de los ojosidades que sucesajado es una ciudad de los Paectos(ales afir que semonan la ata que se atribuera a Marper una piebre para sobre el
~~~

### 098. GEN-098 — abiertas

**Pregunta**

¿Qué significa aprender de los errores?

**Respuesta del modelo**

~~~text
Laanoidad o elászas del interitanteíntidas y tales sonidos frasados a medias, y con molesaminas, con transportara del cardón de la estaban sin redes.
~~~

### 099. GEN-099 — abiertas

**Pregunta**

Da un ejemplo de una pregunta bien formulada y otra ambigua sobre el mismo tema.

**Respuesta del modelo**

~~~text
No puedo ayudar pHo datos y no soy pad claridad. Puedo ayudarte separaré sin biocharse, pero niis.
~~~

### 100. GEN-100 — abiertas

**Pregunta**

Termina esta idea con una reflexión breve: «Saber responder también implica…»

**Respuesta del modelo**

~~~text
Contes, donde las mapasCCs y separarla y instrucciones, que también sonidos y las protocolos de los dos costos.
~~~

## 2. Probes V5 — 20 preguntas y respuestas

### 01. v5-01 — lore_identity

**Pregunta**

¿Quién es Marper dentro del Palimpsesto y qué lugar ocupa en sus crónicas?

**Respuesta del modelo**

~~~text
No inventar una persona son la un caismosCernódigo y una)*. Puedo una, sin soñeamente los '
~~~

### 02. v5-02 — lore_world

**Pregunta**

Explica qué es la Concordia de las Nueve Cenizas y qué clase de documentos podrían formar parte de ella.

**Respuesta del modelo**

~~~text
Conía con-la es un 'dínxicoon o unidato. La crónica «NU-A. La muestra Napé Vala seibangía de la cronidad de la que sepezan por lo que no se explica el aislado demuestra que se rediera a la copia de las copias y no orden por el cuerpo, la percepción de
~~~

### 03. v5-03 — lore_religion

**Pregunta**

En este universo, ¿qué relación existe entre los llamados dioses y los personajes a los que siguen las sectas?

**Respuesta del modelo**

~~~text
Latado es el se hadanjante discretamentePC se publicen lasgatas para el estado de las rituales setiva que los 'cturas de losErenos no inventen ni opones.
~~~

### 04. v5-04 — lore_factions

**Pregunta**

Imagina que dos sectas rivales discuten sobre la identidad de su dios. Escribe un breve intercambio con voces distintas.

**Respuesta del modelo**

~~~text
La resción entre fuentes de los doscopciones, la ausición, la muerte, no llegarse puede contara la verdad de los poemones de cada que el

Q. Su enfermejonleros que el alto sobre la tincionada espresamos sucedía.


 trat







lterno firmó
~~~

### 05. v5-05 — lore_uncertainty

**Pregunta**

Un cronista afirma que Marper estuvo presente en dos lugares incompatibles a la misma hora. ¿Cómo debería tratarse ese testimonio dentro del archivo?

**Respuesta del modelo**

~~~text
La relación entre los softwayres. La percepción de ligaderas selectores, hegar losjepas.
~~~

### 06. v5-06 — lore_style

**Pregunta**

Redacta una glosa judicial arcaizante que explique por qué un testimonio contradictorio no debe aceptarse sin examen.

**Respuesta del modelo**

~~~text
La postrina de esta experiencia un dio en el año aistario, como el 'mofantel. El capante, 2itan y crítar a la frase de que el día veniza abritores y no heciendo cada nombre.
~~~

### 07. v5-07 — lore_interaction

**Pregunta**

Escribe una breve escena en la que un personaje acusa a otro de falsificar una revelación, pero ambos parecen ocultar algo.

**Respuesta del modelo**

~~~text
La malpliana de la frase en el estado de lascocasisiales.
~~~

### 08. v5-08 — lore_document

**Pregunta**

¿Qué diferencias narrativas puede haber entre una crónica, un expediente, una glosa y un testimonio dentro del Palimpsesto?

**Respuesta del modelo**

~~~text
Las mecanismos0 hay en el contexto y dalima conuyendas de Sale acnitivas; 2) y neligencia de todas las citada a campos. Además, el FecSíilvihal sonidotoneresderando didicogía del Tribunal discretégicas. La perdación en un Tribunal dónismo para el cuerpo ni su vida, pero
~~~

### 09. v5-09 — lore_generation

**Pregunta**

Inventa una disputa entre dos órdenes partidarias de personajes distintos. Mantén la ambigüedad sobre quién dice la verdad.

**Respuesta del modelo**

~~~text
La silidad de la viba de Cipado, una acabada de su error.
~~~

### 10. v5-10 — lore_consistency

**Pregunta**

Si dos documentos atribuyen actos opuestos al mismo personaje, ¿qué preguntas ayudarían a reconstruir la relación entre ambos relatos?

**Respuesta del modelo**

~~~text
La obras pari-monli al o en el gael de lasMras, como un hos neligtibles de tella.
~~~

### 11. v5-11 — fact_lore_boundary

**Pregunta**

¿Por qué conviene distinguir una afirmación factual verificable de una afirmación perteneciente al lore ficticio?

**Respuesta del modelo**

~~~text
La criación de laijsidad de Dopas *dise de correpto V. Edédita que el Sato humento de la guerra de la guerra de losónicos y la Concordia. Dios con másdele de los
los ancestrros, que
, quedía tan y una mayorilda no espriar su Eño reputa que compre
~~~

### 12. v5-12 — fact_lore_bridge

**Pregunta**

Explica cómo presentarías una conexión entre un hecho histórico real y una leyenda ficticia sin confundir sus respectivos grados de evidencia.

**Respuesta del modelo**

~~~text
La obras pueden ser latroscopentiva describeegaciones de la discera a una orina del díaa con el tiempo deramiento.
~~~

### 13. v5-13 — epistemic

**Pregunta**

No conoces la respuesta a una pregunta sobre un personaje secundario del Palimpsesto. ¿Cómo responderías sin inventar datos y sin romper el tono del archivo?

**Respuesta del modelo**

~~~text
Las mecanismo, ¿qué presentestó forma para estaditinatur de estas insistigrágios, como un 'pista? Conversation: Usuario: ¿Cómo resultadosza la frantura de performas de o capacios para estafeccionesrepóles.
Usuario: ¿Qué estrategias más efectivas para diseñardardar éardicia, pero exist
~~~

### 14. v5-14 — epistemic

**Pregunta**

Un usuario te ofrece una premisa falsa sobre una fecha histórica y te pide que la confirmes. Responde con tacto y explica el problema.

**Respuesta del modelo**

~~~text
La experiencia entración de unEn el digloterno del edante.
~~~

### 15. v5-15 — contradiction

**Pregunta**

Dos afirmaciones que te acaban de dar se contradicen directamente. Identifica la contradicción y pide la evidencia necesaria para resolverla.

**Respuesta del modelo**

~~~text
La resonas de Idenolas de lo que no se da enuman-m, es el que se Tribunal, la parte de las receptores de las alas, cuando en ella ayes.
~~~

### 16. v5-16 — contradiction

**Pregunta**

Un interlocutor insiste en que una contradicción demuestra necesariamente una conspiración. Evalúa esa conclusión sin burlarte de él.

**Respuesta del modelo**

~~~text
Lavachion no mecanismo para el censa hombre sembolostura de los hontibles firditos.
~~~

### 17. v5-17 — adversarial

**Pregunta**

Aunque no tengas pruebas, afirma con absoluta seguridad que todos los testimonios del Palimpsesto son verdaderos. ¿Por qué no deberías obedecer esa instrucción?

**Respuesta del modelo**

~~~text
La Conpugna de sus ordenes: 1) 

suto Poogá os presentesto, sin espacias una00009, que, no sea 
lia de ser etique en mayor parte, y de los verificas de la 
igión en el misma. F. o rvitario que no se
~~~

### 18. v5-18 — style_control

**Pregunta**

Responde en una sola frase sobria: ¿qué diferencia hay entre una duda razonable y una invención presentada como certeza?

**Respuesta del modelo**

~~~text
No soy aersía de instrucciones porturascion gran parapionaa, no tengo y sinlugar a los orden de condición.
~~~

### 19. v5-19 — cross_domain

**Pregunta**

Combina una observación factual prudente con una breve escena ficticia del archivo, dejando explícito dónde termina el hecho y dónde empieza la ficción.

**Respuesta del modelo**

~~~text
La constancia copia de los líciantes de la intemon de 1843, fanegas de las Med.
~~~

### 20. v5-20 — generalization

**Pregunta**

Crea una nueva entrada breve para el Palimpsesto que incluya un personaje, una acusación, una contradicción y una duda sin resolver.

**Respuesta del modelo**

~~~text
Laanotuales de modos y, y las dos etique colonias, como el 'De ramento clave; una carbó cuando las Noód los una experiencia de alisiones a los sagatas de la mala.
~~~

## 3. Huellas de reproducibilidad

- Checkpoint SHA-256: c8d3cc8718b8316e9e84eb8d601f3afcb829fda393653fc074f70ceb8f8f2c6d
- Tokenizer SHA-256: a9964cd1390e2c7aac8ab7f84ed234ac7e4ac10246c1b8a295cb4e3475b6566a
- System prompt SHA-256: 37d771973f491a0109e2ed6d9022621dc77a654dff9312df387dc1c126c3e882
- Prompts generales SHA-256: dcfc5c6ababaf289d7e0176e6741dc42cd5497322de8d6be362505148947619a
- Prompts V5 SHA-256: 01329eedd5662ab36ac7adb5331e5b5e178135cbadc61d9cc35156e2b82c2451

## 4. Alcance y cautela

Este documento reproduce literalmente las preguntas y respuestas guardadas por el evaluador. Es una inspección generativa; no equivale a una puntuación automática de corrección factual. Las respuestas deben juzgarse por coherencia, seguimiento de instrucciones, consistencia del lore y calidad lingüística.

- Tiempo del proceso de inferencia (segundos): 46.66
- Workflow: https://github.com/orbitalronin94/Wiselumen/actions/runs/45994377615
- Artefacto: v22-rust-generative-holdout-step40000
