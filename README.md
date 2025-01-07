# Trabajo Final de Máster UOC

- Alumno: Alejandro Garcia Herrera
- Tutor: José Luis Iglesias Allones
- Area: 2

## LLMs y flujos RAG - Uso de modelos de lenguaje para interactuar con PDFs

En este trabajo se explora el uso de LLMs para extraer información de documentos PDF. El proyecto se centra en el desarrollo de una aplicación en Python que permita la carga de archivos PDF, la extracción de datos estructurados, y la interacción mediante un chatbot que responda a preguntas abiertas sobre la información de los documentos.

***

## Uso de la aplicación

### 1. Descarga del repositorio y creación del archivo de entorno.

En primer lugar, se debe clonar el repositorio a un directorio local con el comando de bash

`git clone https://github.com/Narkal2/tfm_garciaherrera.git directorio_destino`

O bien descargar y descomprimir el .zip

Después, se debe crear un fichero llamado `.env` en la raíz del repositorio, que contenga los siguientes campos:

```
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
OPENAI_API_KEY = "YOUR_GPT_API_KEY"
ENVIRONMENT = "local"
REDIS_URL_LOCAL = "redis://localhost:6379"
REDIS_URL_DOCKER = "redis://redis:6379"
REDIS_INDEX = "documents"
EMB_MODELS_NAME = "sentence-transformers/all-mpnet-base-v2"
```

Para el uso de las APIs de los LLMs de Google y OpenAI se deben introducir las API Keys correspondientes, que no se han subido al repositorio por cuestiones de seguridad.

### 2. Levantar la aplicación con `docker-compose`

Para crear la imagen de docker y levantar los contenedores de los diferentes servicios se debe disponer de un motor de Docker instalado en el equipo. La sintaxis de `docker-compose` permite definir los diferentes contenedores en un mismo archivo, en el que se especifica la configuración de los mismos. La aplicación se levanta desde el directorio raíz del repositorio con el siguiente comando:

`docker compose -f "docker-compose.yml" up -d --build `

Una vez terminada de usar, para apagar los servicios hay que utilizar el siguiente comando, también desde el directorio raíz:

`docker compose -f "docker-compose.yml" down`

### 3. Conexión al Frontend

Con los contenedores levantados, para acceder a la aplicación se debe ir a la URL del frontend:

`http://localhost:8030`

***

## Videos de demostración de la aplicación

### Extracción de información

https://github.com/user-attachments/assets/d2257098-baf0-484e-9f2d-86a30246adfa

### Asistente conversacional

https://github.com/user-attachments/assets/dac16a4b-0765-4d25-ba45-36c1db827ae0


