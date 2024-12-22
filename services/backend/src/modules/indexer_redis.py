from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Redis
from pathlib import Path
from dotenv import load_dotenv
import re

class DocumentIndexer:
    def __init__(self, redis_url: str, index_name: str, embeddings):
        """
        Inicializa el indexador de documentos.

        Args:
            - redis_url: URL del servidor Redis.
            - index_name: Nombre del índice en Redis.
            - embeddings: Modelo de embeddings utilizado para procesar los documentos.
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.embeddings = embeddings
        self.vectorstore = None

    def _check_index_exists(self) -> bool:
        """
        Verifica si el índice existe en Redis.

        Returns:
            - True si el índice existe, False en caso contrario.
        """
        try:
            client = self.vectorstore.client
            client.ft(self.index_name).info()
        except:
            return False
        return True

    def _clean_text(self, text: str) -> str:
        """
        Escapa caracteres especiales en los metadatos.

        Args:
            - text: Texto a procesar.

        Returns:
            - Texto con caracteres especiales escapados.
        """
        return re.sub(r'[.,<>{}/[\]"\':;!¡?¿@#$€%^&*ªº·()\-+=~ ]', r'\\\g<0>', text)

    def index_documents_redis(self, files: list, loader_name: str='PyMuPDF') -> dict:
        """
        Procesa e indexa una lista de archivos PDF en Redis.

        Args:
            - files: Lista de rutas de archivos a indexar.
            - loader_name: Nombre del cargador de documentos (por defecto 'PyMuPDF').

        Returns:
            - Diccionario con el resultado del proceso de indexación.
        """
        if not files:
            return {"success": False, "message": "No se proporcionaron archivos para indexar."}
        
        # Eliminar índice existente
        try:
            if self._check_index_exists():
                client = Redis(
                    redis_url=self.redis_url,
                    index_name=self.index_name,
                    embedding=self.embeddings
                ).client
                client.ft(self.index_name).dropindex(delete_documents=True)
        except Exception as e:
            return {"success": False, "message": f"Error al eliminar el índice existente: {e}"}

        documents = []
        for file in files:
            try:
                if loader_name == 'PyMuPDF':
                    loader = PyMuPDFLoader(file)
                    pages = loader.load_and_split()
                else:
                    # Add LLMSherpa Loader
                    ...
                for chunk in pages:
                    metadata = {
                        "page": chunk.metadata.get("page", 0),
                        "filename": self._clean_text(Path(file).name),
                        "filepath": self._clean_text(file),
                    }
                    document = Document(page_content=chunk.page_content, metadata=metadata)
                    documents.append(document)
            except Exception as e:
                return {"success": False, "message": f"Error al procesar {file}: {e}"}

        # Crear nuevo índice en Redis
        try:
            self.vectorstore = Redis.from_documents(
                documents,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=self.index_name,
            )
        except Exception as e:
            return {"success": False, "message": f"Error al indexar los documentos: {e}"}

        return {"success": True, "message": f"Se indexaron {len(documents)} documentos."}


if __name__ == "__main__":
    """
    Ejemplo de uso de la clase DocumentIndexer.
    """
    load_dotenv()
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    redis_url = "redis://localhost:6379"
    index_name = "documents"
    document_indexer = DocumentIndexer(redis_url, index_name, embeddings_model)

    file_path = ["ES-00001.pdf"]
    result = document_indexer.index_documents_redis(file_path)
    print(result)

    retriever_pdf = document_indexer.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        })
    relevant_docs = retriever_pdf.get_relevant_documents(
        query="Cual es la función del producto?"
    )
    #print(relevant_docs)