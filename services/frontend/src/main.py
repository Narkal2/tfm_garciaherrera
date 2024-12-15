import streamlit as st
#from streamlit_theme import st_theme
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()

env = os.getenv('ENVIRONMENT', 'local')

if env == 'docker':
    FASTAPI_URL = "http://backend:8000"
else:
    FASTAPI_URL = "http://localhost:8000"


def send_text(query, llm_type, temperature, num_chunks):
    """
    Envía una consulta de texto al backend para obtener una respuesta generada por el modelo LLM.

    Args:
        query (str): Pregunta o consulta realizada por el usuario.
        llm_type (str): Tipo de modelo LLM a utilizar.
        temperature (float): Valor de temperatura para ajustar la creatividad del modelo.
        num_chunks (int): Número de fragmentos de contexto que se usarán en la llamada.

    Returns:
        dict: Respuesta del modelo en formato JSON.
    """
    payload = {
        "query": query,
        "llm_type": llm_type,
        "temperature": temperature,
        "num_chunks": num_chunks,
    }
    response = requests.post(f"{FASTAPI_URL}/query", data=payload)
    return response.json()


def upload_files_to_vectorstore(files):
    """
    Envía archivos PDF al backend para que sean indexados en el vectorstore.

    Args:
        files (list): Lista de archivos PDF cargados por el usuario.

    Returns:
        list: Nombres de los archivos que fueron subidos exitosamente al backend.
    """
    uploaded_names = []
    files_payload = [("files", (file.name, file.read(), "application/pdf")) for file in files]

    try:
        # Enviar los archivos al backend
        response = requests.post(f"{FASTAPI_URL}/index_files", files=files_payload)
        if response.status_code == 200:
            uploaded_names = [file.name for file in files]
        else:
            st.error(f"Error al indexar los archivos: {response.json().get('detail', 'Error desconocido')}")
    except requests.RequestException as e:
        st.error(f"Error de red al indexar los archivos: {e}")

    return uploaded_names


def extract_information(pdf_file, json_file=None):
    """
    Envía un archivo PDF y, opcionalmente, un esquema JSON al backend para la extracción de información estructurada.

    Args:
        pdf_file (File): Archivo PDF cargado por el usuario.
        json_file (File, optional): Esquema JSON que define la estructura esperada de los datos.

    Returns:
        dict: Datos estructurados extraídos del PDF en formato JSON, o `None` en caso de error.
    """
    files_payload = {
        "pdf_file": (pdf_file.name, pdf_file.read(), "application/pdf")
    }
    if json_file:
        files_payload["json_schema"] = (json_file.name, json_file.read(), "application/json")

    try:
        response = requests.post(f"{FASTAPI_URL}/extract_information", files=files_payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error del servidor: {response.status_code} - {response.json().get('detail', 'Error desconocido')}")
            return None
    except requests.RequestException as e:
        st.error(f"Error al conectar con el backend: {e}")
        return None


def main():
    """
    Función principal que contiene la lógica del Frontend de Streamlit
    """
    st.set_page_config(
        page_title="TFM - Alejandro García Herrera",
        page_icon=str(Path(__file__).parent.parent / "images" / "LogoUOC.png")
    )
    
    if "mode" not in st.session_state:
        st.session_state["mode"] = None

    st.sidebar.title("Menú Principal")

    # Configuración de botones para seleccionar la funcionalidad de la aplicación
    left, right = st.sidebar.columns(2)
    if left.button("Asistente conversacional", use_container_width=True):
        st.session_state["mode"] = "Chatbot"
    if right.button("Extracción de información", use_container_width=True):
        st.session_state["mode"] = "Extracción Información"

    ######################### CHATBOT ############################
    if st.session_state["mode"] == "Chatbot":
        st.sidebar.subheader("Configuración del asistente")

        # Selección del modelo LLM
        llm_type = st.sidebar.radio("Seleccione el modelo LLM:", ["gemini-pro", "gpt-4o-mini", "otro-llm"])
        # Configuración de temperatura
        temperature = st.sidebar.number_input(
            "Temperatura (entre 0 y 1):",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            format="%.1f"
        )
        # Configuración del número de chunks
        num_chunks = st.sidebar.number_input(
            "Número de chunks recuperados (entre 1 y 10):",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        # Subida de múltiples archivos para la indexación
        uploaded_files = st.sidebar.file_uploader(
            "Subir archivos PDF para el chatbot:",
            type="pdf",
            accept_multiple_files=True
        )

        if st.sidebar.button("Indexar documentos"):
            if uploaded_files:
                with st.spinner("Indexando documentos..."):
                    uploaded_names = upload_files_to_vectorstore(uploaded_files)
                    if uploaded_names:
                        st.sidebar.success(f"Documentos indexados: {', '.join(uploaded_names)}")
                        st.session_state["documents_indexed"] = True
                        st.session_state.messages = [] # Added to remove chat history when reindexing
                    else:
                        st.sidebar.error("No se pudieron indexar los documentos.")
            else:
                st.sidebar.warning("Por favor, suba al menos un archivo.")
        
        documents_indexed = st.session_state.get("documents_indexed", False)
        st.title("Asistente Conversacional sobre contenido de documentos PDF")
        st.markdown("Realiza consultas basadas en los documentos previamente indexados.")

        if not documents_indexed:
            st.text_input(
                "Escriba su consulta aquí",
                disabled=True,
                placeholder="Debes indexar documentos antes de realizar consultas."
            )
        else:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if query := st.chat_input("Escriba su consulta aquí"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    try:
                        response = send_text(query, llm_type, temperature, num_chunks)
                        result = response.get("bot_response") #, "No se recibió respuesta del backend.")
                    except requests.RequestException as e:
                        result = f"Error al contactar el backend: {e}"
                    st.markdown(result)

                st.session_state.messages.append({"role": "assistant", "content": result})


    ######################### EXTRACCIÓN INFORMACIÓN ############################
    elif st.session_state["mode"] == "Extracción Información":

        st.title("Extracción de Información")

        st.sidebar.subheader("Subida de archivo para extracción de información")

        schema_option = st.sidebar.radio(
            "Seleccione la opción de extracción:",
            ["Adjuntar esquema JSON", "Inferir esquema automáticamente"]
        )

        # Subida de archivos basada en la elección del usuario
        if schema_option == "Adjuntar esquema JSON":
            st.sidebar.write("Por favor, suba los archivos necesarios:")
            uploaded_pdf = st.sidebar.file_uploader(
                "Subir archivo PDF:",
                type="pdf",
                accept_multiple_files=False
            )
            uploaded_json = st.sidebar.file_uploader(
                "Subir esquema JSON:",
                type="json",
                accept_multiple_files=False
            )
        else:
            st.sidebar.write("Por favor, suba un archivo PDF:")
            uploaded_pdf = st.sidebar.file_uploader(
                "Subir archivo PDF:",
                type="pdf",
                accept_multiple_files=False
            )
            uploaded_json = None

        # Botón para procesar la solicitud
        if st.sidebar.button("Procesar archivo"):
            if uploaded_pdf:
                with st.spinner("Procesando archivo..."):
                    # Llamada al backend para extraer la información
                    extraction_result = extract_information(uploaded_pdf, uploaded_json)
                    
                    if extraction_result:
                        st.success("Procesamiento completado con éxito.")
                        
                        # Guardar el resultado en el estado de la sesión
                        st.session_state["extraction_result"] = extraction_result
                    else:
                        st.error("No se pudo procesar el archivo. Verifique los datos.")
            else:
                st.sidebar.warning("Por favor, suba el archivo PDF necesario.")

        if "extraction_result" in st.session_state and st.session_state["extraction_result"]:
            formatted_json = json.dumps(st.session_state["extraction_result"], indent=4, ensure_ascii=False)
            st.code(formatted_json, language="json")
            # Boton para descargar el archivo en formato JSON
            st.download_button(
                label="Descargar JSON",
                data=str(formatted_json),
                file_name=f"{str(uploaded_pdf.name).replace('.pdf','')}_result.json",
                mime="application/json"
            )

        else:
            st.write("Aquí se mostrará la información extraída en formato JSON una vez procesada.")

    else:
        st.sidebar.write("Por favor, seleccione una funcionalidad.")

if __name__ == "__main__":
    main()

    # USO: Con el FastAPI levantado, ejecutar "streamlit run main.py" en terminal