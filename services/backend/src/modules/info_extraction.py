import os
import json
from dotenv import load_dotenv
import pymupdf4llm
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#import vertexai
# from vertexai.preview.generative_models import GenerativeModel
# from vertexai.preview import tokenization
from typing import Dict, Any

load_dotenv()


# def extract_json_from_response(response):
#     """
#     Extrae el JSON formateado desde la respuesta del modelo.
    
#     Args:
#         response (GenerationResponse): Respuesta del modelo.

#     Returns:
#         dict: Un diccionario con el contenido extraído.
#     """
#     try:
#         if not response.candidates:
#             raise ValueError("La respuesta no contiene candidatos.")

#         candidate_content = response.candidates[0].content
#         if not candidate_content.parts:
#             raise ValueError("El contenido del candidato no tiene partes.")

#         candidate_text = candidate_content.parts[0].text
#         if not candidate_text:
#             raise ValueError("El texto candidato está vacío.")

#         candidate_text = candidate_text.replace("\n", '')
#         candidate_text = candidate_text.replace("```json", '')
#         candidate_text = candidate_text.replace("```", '')

#         return json.loads(candidate_text)

#     except (ValueError, json.JSONDecodeError) as e:
#         return f"Error al procesar la respuesta de Gemini: {e}"


class PDFProcessor:
    def __init__(self, llm_type: str):
        """
        Initialize PDF processor
        """
        self.llm_type = llm_type
        self.llm = None

    def init_llm(self, **llm_kwargs):
        """
        Inicializa el modelo LLM seleccionado dinámicamente.
        
        Args:
        - llm_type: str, tipo de modelo LLM ("google-genai" o "azure-openai").
        - llm_kwargs: dict, parámetros adicionales para la configuración del modelo.
        """
        if self.llm_type == "gemini-pro":
            self.llm = ChatGoogleGenerativeAI(
                model=llm_kwargs.get("model", "gemini-pro"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=llm_kwargs.get("temperature", 0)
            )
        elif self.llm_type == "gpt-4o-mini":
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key = os.getenv("OPENAI_API_KEY"),
                temperature=llm_kwargs.get("temperature", 0),
                max_tokens=4096,
            )
        else:
            raise ValueError(f"Modelo LLM no soportado: {self.llm_type}")

    def process_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using pymupdf4llm.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text in markdown format.
        """
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            return md_text
        except Exception as e:
            raise ValueError(f"Error al procesar el PDF: {e}")

    def generate_prompt(self, extracted_text: str, schema: Dict[str, Any]) -> str:
        """
        Generate the prompt for the LLM call.

        Args:
            extracted_text (str): Text extracted from the PDF.
            schema (Dict[str, Any]): Schema in JSON format to guide the output.

        Returns:
            str: Generated prompt.
        """
        return f"""
        Eres un experto en análisis de documentos.  
        Se te proporciona el texto extraído de un documento en formato markdown y un esquema JSON.  
        Tu tarea es analizar el contenido del documento y completarlo en el formato estructurado que indica el esquema JSON proporcionado.  

        Aquí está el esquema JSON:  
        {json.dumps(schema, indent=2, ensure_ascii=False)}  

        Aquí está el texto extraído del documento:  

        {extracted_text}

        Por favor, devuelve únicamente el JSON estructurado en el formato descrito en el esquema proporcionado.
        """

    def generate_output(self, pdf_path: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured output for the given PDF based on the schema.

        Args:
            pdf_path (str): Path to the PDF file.
            schema (Dict[str, Any]): Schema in JSON format.

        Returns:
            dict: Generated structured output.
        """
        try:
            extracted_text = self.process_pdf(pdf_path)
            prompt = self.generate_prompt(extracted_text, schema)
            
            # Validate schema function names
            # for key in schema.keys():
            #     if not self.is_valid_function_name(key):
            #         raise ValueError(f"Invalid function name in schema: {key}")

            if self.llm_type == "gpt-4o-mini":
                structured_llm = self.llm.with_structured_output(schema=schema, method="json_mode")
                response = structured_llm.invoke(prompt)
                return response
            
            ## NO FUNCIONA!! Implementar en el frontend la parte de GPT
            # elif self.llm_type == "gemini-pro":
            #     typed_dict_code = convert_json_schema_to_typeddict(schema, "GeminiSchema")
            #     print(typed_dict_code)
            #     structured_llm = self.llm.with_structured_output(schema=typed_dict_code)
            #     response = structured_llm.invoke(prompt)
            #     return response
            #     response = self.llm.generate_content(prompt)
            #     results = extract_json_from_response(response)
            #     return results
        
        except Exception as e:
            print(f"Error al generar la salida estructurada: {e}")
            return {}

    # def is_valid_function_name(self, name: str) -> bool:
    #     import re
    #     return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]{0,63}$', name))

    def save_output(self, results: dict, output_file: str):
        """
        Save structured output in JSON format.

        Args:
            results (dict): Generated structured output.
            output_file (str): Path to save the JSON file.
        """
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    """
    Ejemplo de uso de la clase PDFProcessor.
    """
    from pathlib import Path
    pdf_file_path = Path(__file__).parent / "ES-00001.pdf"
    schema_file_path = Path(__file__).parent / "fitosanitario_schema.json"
    output_file_path = Path(__file__).parent / "output_gemini.json"

    processor = PDFProcessor(
        llm_type="gpt-4o-mini"
        #llm_type="gemini-pro"
        )

    processor.init_llm(
        #llm_type="gemini-pro",
        #llm_type="gpt-4o-mini",
        temperature=0
    )

    try:
        with open(schema_file_path, "r") as schema_file:
            schema = json.load(schema_file)
        results = processor.generate_output(pdf_path=pdf_file_path, schema=schema)
        if results:
            processor.save_output(results, output_file_path)
            print(f"Output generado y guardado en: {output_file_path}")
        else:
            print("No se generaron resultados.")

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
    except json.JSONDecodeError as e:
        print(f"Error al leer el archivo de esquema JSON: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")