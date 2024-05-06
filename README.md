# Modelo de Scoring de Crédito con Machine Learning
>Este documento presenta la decripción de la solución, la arquitectura y las principales consideraciones y pasos requeridos para realizar el despliegue e instalación del Modelo de Scoring de Crédito con Machine Learning.

Tambien, en los siguientes links se encuentra la informacion documental asociada al proyecto:

Carpeta Asociada asociada a Protocolo de Informacion de Equipo:[https://livejaverianaedu.sharepoint.com/:f:/r/sites/PruebasdeconceptoacadmicasCAOBA-poc-069-lumon-riesgo-crediticio-fase1/Shared%20Documents/poc-069-lumon-riesgo-crediticio-fase1/1_Proyecto%20y%20seguimiento?csf=1&web=1&e=ZG3kVP]

Informe Proyecto: [https://livejaverianaedu.sharepoint.com/:w:/r/sites/PruebasdeconceptoacadmicasCAOBA-poc-069-lumon-riesgo-crediticio-fase1/Shared%20Documents/poc-069-lumon-riesgo-crediticio-fase1/1_Proyecto%20y%20seguimiento/Informe%20proyecto/1_Propuesta%20de%20proyecto.docx?d=w7026f59589424f289fa3cc23861e7881&csf=1&web=1&e=FYgOyc]

Manual de Usuario: (**Pendiente**)

Video Demo: (**Pediente**)

Plantilla de Comunicaciones:[(https://livejaverianaedu.sharepoint.com/:w:/r/sites/PruebasdeconceptoacadmicasCAOBA-poc-069-lumon-riesgo-crediticio-fase1/Shared%20Documents/poc-069-lumon-riesgo-crediticio-fase1/1_Proyecto%20y%20seguimiento/Plantillas%20Caoba/PLANTILLA%20DE%20INSUMOS%20_CAOBA_PoC%20069.docx?d=w7c1b8ef32d9f480f996d6934cf14d5ea&csf=1&web=1&e=DNyObh)] 

## Tabla de Contenidos
* [Descripción de la solución](#descripción-de-la-solución)
* [Screenshots](#screenshots)
* [Requerimientos](#requerimientos)
* [Instalacion](#instalación)
* [Ejemplos de Codigo](#ejemplos-de-codigo)
* [Pruebas Automatizadas](#pruebas-automatizadas)
* [Autores](#autores)

## Descripción de la solución
 En Lumon SAS para continuar con su línea de negocio de Fintech se requiere una herramienta para el análisis de riesgo financiero de los clientes a los que se adjudica los créditos, por lo tanto, se quiere aprovechar la bases de datos que hoy se tiene de los numerosos prestamos desembolsados a los usuarios para desarrollar un algoritmo predictivo de machine learning que genere la probabilidad de impago del cliente, generando así una adjudicación optima de los créditos, garantizando los flujos de caja desembolsados de los créditos y reduciendo los costos en casas de cobranza

### Reto del cliente
LUMON desarrolla software para el control de riesgo financiero. En el funcionamiento de esta línea de negocio se ha evidenciado la continua presencia del riesgo financiero para nuestros clientes, en temas de impagos de los créditos, incurriendo en costos de cobranza para recuperar la cartera vencida.
Por tanto, se plantea la pregunta: ¿Cuál es la probabilidad que el usuario no pague el crédito recibido?
### Solución Alianza CAOBA
En la primera reunión sostenida con la empresa se aclaró la expectativa y alcance del proyecto, definiendo como entregable final un modelo de Machine Learning desarrollado en Python para predecir la probabilidad de impago del cliente.
### Impacto potencial esperado en el Negocio
1. Reducir la incidencia de incumplimientos de préstamos entre los usuarios de nuestra plataforma fintech.
2. Reducir los costos de cobranza identificando de forma preventiva los préstamos en riesgo.
3. Mejorar la retención de clientes ofreciendo intervenciones oportunas para los usuarios en riesgo.
4. Integrar insights predictivos en los procesos de aprobación y monitoreo de préstamos de los clientes
5. Tener una cartera más sana, conformada por clientes con buenos puntajes crediticios y una baja posibilidad de default

**Descripción de la solución**
(**Pediente imagen**)

### Screenshots / Demo
(**Pediente demo**)

## Arquitectura logica de la solución

**Preprocesamiento:**
El proceso inicia con el dataset entregado por Lumon SAS, el cual contiene información socioeconómica e información crediticia de los usuarios con créditos, donde se realiza un análisis exploratorio y descriptivo de cada una de las columnas con el objetivo de entender las variables entregadas. Posteriormente se implementa un proceso de limpieza y selección de características del dataset donde se decide eliminar ciertas variables de la base de datos de acuerdo con la conversación con el equipo Lumon y los resultados de los análisis exploratorios donde se visualizan columnas repetidas, columnas sin varianza (datos con valor único), columnas linealmente dependendientes, columnas altamente relacionadas o columnas con datos no agregan valor al módelo. 

Este proceso genera como resultado dos dataset: 
* [DB sin valores nulos:] Esta base de datos tiene los datos socioeconómicos e información de crédito de los usuarios. El porcentaje total de usuarios que diligenciaron la información socioeconómica es del 3% por ello se decidió separar esta información para modelarla por separado.
* [DB sin información de usuarios:] Este dataset comprende solo las columnas con información asociada a los créditos incluyendo la totalidad de los registros.

Esta decisión de toma para aprovechar la data entregada y no eliminar el 97% de la información al no tener datos de la información de usuarios.

A la base de datos con información de los usuarios se le implementaran procesos de NLP para estandarizar valores de las columnas y reducir la dispersión del dataset. Luego se hará un proceso de balanceo de datos con la base de datos obtenida del proceso de estandarización de texto y el dataset sin información de usuarios, todo esto con el fin implementar dos modelos con cada base de datos y luego hacer la integración de estos. 
 
 **Modelamiento:**

**Diagrama** 
![](structure_example/docs/readme/docs_DiagramaArquitectura.png)



## Estructura del proyecto

```
.
├── README.md
├── data/
│   ├── raw/
│   │   ├── db_v0.xlsx
│   └── stage/
│       └── db_stage_infousers.cvs
│       └── db_stage_reducida.cvs
│   ├── analytics/
├── datalab/
    ├──EDA.ipynb
    ├──Cleannig.ipynb
    ├──basicdescriptives_mod.py
├── src/
├── conf/
├── docs/
│   └── readme/
│       ├── docs_DiagramaArquitectura.png
├── dashboard/
├── deploy/
└── temp/
    ├──basicdescriptives.py
```


## Proceso de ejecucion y despliegue

## Requerimientos
**Nota:** Obligatorio: Minimo debe escribir los requerimientos por cada lenguaje de programacion usado tanto en el back-end (Ej: Python, R) como en el front-end, si aplica. Tambien, es importante que ponga las versiones correspondientes 
### Librerias Empleadas 
Para llenar
### Requerimientos Hardware
Para llenar
### Requerimientos Software
Para llenar

## Instalación: 
**Nota:** Obligatorio: Minimo debe haber en el proyecto el archivo que permita instalar el ambiente necesario para el despliegue de la solución y los comandos ejecutados para la instalacion. Por ejemplo, si es Python un requeriments.txt o un archivo de DESCRIPTION en R. 

## Configuracion
**Nota:** Para llenar

## Ejemplos de Codigo
**Nota:** Para llenar

## Errores conocidos
**Nota:** Para llenar

## Pruebas Automatizadas
**Nota:** Si aplica puede poner como correr las pruebas

## Imagenes
**Nota:** Si aplica puede poner cuales fueron las imagenes usadas (Ejemplo: Docker)

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Autores
**Nota:** Obligatorio: Minimo debe llenar los autores tanto de analitica como del negocio,su organizacion, su nombre con el nombre del papel que tomo en el equipo, su respectivo correo electronico

| Organización   | Nombre del Miembro | Correo electronico | 
|----------|-------------|-------------|
| PUJ-Bogota |  Persona 1: Cientific@ de Datos | ejemplo@XXXX |
| Organizacion  |  Persona 2:Lider del negocio  | ejemplo@XXXX |


