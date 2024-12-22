import os
import warnings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")


class Assistant:
    def __init__(self, llm_type: str, retriever):
        """
        Inicializa el asistente con un retriever que interactúa con el vectorstore.

        Args:
            - retriever: Objeto utilizado para recuperar documentos relevantes del vectorstore.
        """
        self.llm_type = llm_type
        self.retriever = retriever
        self.llm = None
        self.rag_chain = None

    def init_llm(self, **llm_kwargs):
        """
        Inicializa el modelo LLM seleccionado dinámicamente.

        Args:
            - llm_type: Tipo de modelo LLM.
            - llm_kwargs: Parámetros adicionales para la configuración del modelo.
        """
        if self.llm_type == "Gemini-1.5-flash":
            self.llm = GoogleGenerativeAI(
                model=llm_kwargs.get("model", "gemini-1.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=llm_kwargs.get("temperature", 0.3)
            )

        elif self.llm_type == "GPT-4o-mini":
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key = os.getenv("OPENAI_API_KEY"),
                temperature=llm_kwargs.get("temperature", 0.3),
                max_tokens=1000,
            )

        elif self.llm_type == "Llama-3.3":
            model = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                temperature=llm_kwargs.get("temperature", 0.3),
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )
            self.llm = ChatHuggingFace(llm=model)

        else:
            raise ValueError(f"Modelo LLM no soportado: {self.llm_type}")

    def init_prompt(self):
        """
        Inicializa el prompt utilizado para generar las respuestas.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """Eres un asistente conversacional que debe ayudar al usuario a obtener respuestas a sus preguntas.
                    Utiliza la información del contexto proporcionado para responder a las preguntas del usuario.
                    
                    Question: {question} 
                    Context: {context} 
                    Answer:""",
                ),
            ]
        )

    @staticmethod
    def format_docs(docs):
        """
        Formatea los documentos recuperados para que puedan ser utilizados como contexto en el prompt.

        Args:
            - docs: Lista de documentos.

        Returns:
            - Cadena de texto con el contenido de los fragmentos separados por doble salto de línea.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def init_rag_chain(self):
        """
        Configura la Chain del flujo RAG.
        """
        if not self.llm or not self.prompt:
            raise ValueError("Se deben inicializar el LLM y el prompt antes de configurar la Chain.")
        
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, question):
        """
        Responde a una pregunta usando el LLM y el vectorstore.

        Args:
            - question: Pregunta del usuario.

        Returns:
            - Respuesta generada por el modelo.
        """
        if not self.rag_chain:
            raise ValueError("La cadena RAG no está configurada. Llama a `init_rag_chain` primero.")

        try:
            # Retrieve context using the retriever
            context = self.retriever.invoke(question)
            print("Contexto recuperado:", context)
            response = self.rag_chain.invoke(question, verbose=True)
            return response
        except Exception as e:
            return f"Error al procesar la pregunta: {e}"

if __name__ == "__main__":
    """
    Ejemplo de uso de la clase Assistant.
    """
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Redis

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    redis_url = "redis://localhost:6379"
    index_name = "documents"
    rds = Redis(
                embedding=embeddings_model,
                index_name=index_name,
                redis_url=redis_url,
                )
    retriever = rds.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    #assistant = Assistant(llm_type="Gemini-1.5-flash", retriever=retriever)
    #assistant = Assistant(llm_type="GPT-4o-mini", retriever=retriever)
    assistant = Assistant(llm_type="Llama-3.3", retriever=retriever)

    # Inicializar LLM y cadena RAG
    assistant.init_llm(temperature=0.3)
    print(assistant.llm)
    assistant.init_prompt()
    print(assistant.prompt)
    assistant.init_rag_chain()
    
    query = "¿Que puntos son los más destacados?"

    print(assistant.ask_question(query))
