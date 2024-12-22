from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from typing import Optional
import json

from langchain_community.vectorstores import Redis
from langchain_huggingface import HuggingFaceEmbeddings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.modules.assistant import Assistant
from src.modules.info_extraction import PDFProcessor
from src.modules.indexer_redis import DocumentIndexer

load_dotenv()

app = FastAPI()

# Carga de los parámetros de entorno
env = os.getenv('ENVIRONMENT', 'local')
if env == 'docker':
    redis_url = os.getenv("REDIS_URL_DOCKER")
else:
    redis_url = os.getenv("REDIS_URL_LOCAL")
embeddings_model_name = os.getenv("EMB_MODELS_NAME")
index_name = os.getenv("REDIS_INDEX")

# Inicializar el modelo de embeddings, la conexión con Redis y la clase del indexador
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
document_indexer = DocumentIndexer(redis_url, index_name, embeddings_model)
rds = Redis(
            embedding=embeddings_model,
            index_name=index_name,
            redis_url=redis_url,
            )

@app.post("/query")
async def post_query(
    query: str = Form(...),
    llm_type: str = Form("GPT-4o-mini"),
    temperature: float = Form(0.3),
    num_chunks: int = Form(5)
):
    """
    Procesa preguntas del usuario y genera respuestas utilizando la clase del asistente.

    Args:
        - query: Pregunta del usuario.
        - llm_type: Tipo de modelo de lenguaje.
        - temperature: Valor de temperatura del modelo.
        - num_chunks: Cantidad de fragmentos de información a recuperar.

    Returns:
        JSON con la consulta del usuario y la respuesta generada por el asistente.
    """
    try:
        retriever = rds.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        assistant = Assistant(llm_type=llm_type, retriever=retriever)
        assistant.init_llm(temperature=temperature)
        assistant.init_prompt()
        assistant.init_rag_chain()
        response = assistant.ask_question(query)
        print(response)
        return JSONResponse(content={
            "user_query": query,
            "bot_response": response
        })
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error interno: {e}"})


@app.post("/extract_information")
async def extract_information(
    pdf_file: UploadFile = File(...),
    llm_type: str = Form("GPT-4o-mini"),
    json_schema: Optional[UploadFile] = None
    ):
    """
    Extrae información de un archivo PDF utilizando un modelo de IA generativa y un esquema JSON opcional.

    Args:
        - pdf_file: Archivo PDF cargado por el usuario.
        - json_schema: Esquema JSON opcional para estructurar la salida.

    Returns:
        JSON con la información estructurada extraída del PDF.
    """
    try:
        pdf_path = f"/tmp/{pdf_file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await pdf_file.read())

        schema = {}
        if json_schema:
            schema_content = await json_schema.read()
            schema = json.loads(schema_content)

        processor = PDFProcessor(llm_type=llm_type)
        processor.init_llm(temperature=0)
        results = processor.generate_output(pdf_path=pdf_path, schema=schema)

        os.remove(pdf_path)

        if results:
            return JSONResponse(content=results)
        else:
            return JSONResponse(status_code=500, content={"error": "No se pudo generar la salida estructurada."})

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "El archivo de esquema JSON no es válido."})
    except FileNotFoundError:
        return JSONResponse(status_code=400, content={"error": "Archivo PDF no encontrado."})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Error en el procesamiento: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error interno: {e}"})


@app.post("/index_files")
async def index_files(files: list[UploadFile] = File(...)):
    """
    Indexa múltiples documentos PDF en Redis.

    Args:
        - files: Lista de archivos PDF a indexar.

    Returns:
        JSON indicando el éxito o error de la operación.
    """
    try:
        file_paths = []
        for file in files:
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(temp_path)

        document_indexer.index_documents_redis(file_paths)

        for path in file_paths:
            os.remove(path)

        return JSONResponse(content={"message": "Archivos indexados correctamente."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error al indexar los archivos: {e}"})


if __name__ == "__main__":
    """
    Ejecuta el servicio del Backend.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

