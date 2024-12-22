import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import pymupdf4llm
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import google.generativeai as genai

from typing import Dict, Any

load_dotenv()

def extract_json_from_response(response, model_type):
    """
    Extrae el JSON formateado desde la respuesta del modelo.
    
    Args:
        response (GenerationResponse): Respuesta del modelo.

    Returns:
        dict: Un diccionario con el contenido extraído.
    """
    try:
        if model_type == "Gemini-1.5-flash":
            if not response.candidates:
                raise ValueError("La respuesta no contiene candidatos.")

            candidate_content = response.candidates[0].content
            if not candidate_content.parts:
                raise ValueError("El contenido del candidato no tiene partes.")

            candidate_text = candidate_content.parts[0].text
            if not candidate_text:
                raise ValueError("El texto candidato está vacío.")

            candidate_text = candidate_text.replace("\n", '')
            candidate_text = candidate_text.replace("```json", '')
            candidate_text = candidate_text.replace("```", '')
            candidate_text = ' '.join(candidate_text.split())
            print(candidate_text)
            return json.loads(candidate_text)
        
        elif model_type == "Llama-3.3":
            text = response.content
            text = text.replace("\n", '')
            text = text.replace("```json", '')
            text = text.replace("```", '')
            text = ' '.join(text.split())
            print(text)
            return json.loads(text)

    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error al procesar la respuesta de Gemini/Llama: {e}")
        try:
            if model_type == "Gemini-1.5-flash":
                return candidate_text
            elif model_type == "Llama-3.3":
                return text
        except Exception as error_2:
            return f"Error al procesar la respuesta de Gemini/Llama: {error_2}"


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
        if self.llm_type == "Gemini-1.5-flash":
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.llm = genai.GenerativeModel(model_name='gemini-1.5-flash')

        elif self.llm_type == "GPT-4o-mini":
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key = os.getenv("OPENAI_API_KEY"),
                temperature=llm_kwargs.get("temperature", 0),
                max_tokens=4096,
            )

        elif self.llm_type == "Llama-3.3":
            model = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                temperature=llm_kwargs.get("temperature", 0),
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )
            self.llm = ChatHuggingFace(llm=model)

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
        En caso de que el JSON esté vacío o encuentres nuevos campos, intenta inferir la estructura que debería tener el JSON resultante.

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

            if self.llm_type == "GPT-4o-mini":
                structured_llm = self.llm.with_structured_output(schema=schema, method="json_mode")
                response = structured_llm.invoke(prompt)
                return response
            
            elif self.llm_type == "Gemini-1.5-flash":
                response = self.llm.generate_content(prompt)
                results = extract_json_from_response(response, self.llm_type)
                return results
            
            elif self.llm_type == "Llama-3.3":
                response = self.llm.invoke(prompt)
                results = extract_json_from_response(response, self.llm_type)
                return results
        
        except Exception as e:
            print(f"Error al generar la salida estructurada: {e}")
            return {}

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
    try:
        pdf_file_path = Path(__file__).parent / "ES-00001.pdf"
        schema_file_path = Path(__file__).parent / "fitosanitario_schema.json"
        output_file_path = Path(__file__).parent / "output.json"
        schemaless = False

        processor = PDFProcessor(llm_type="Gemini-1.5-flash")
        processor.init_llm(temperature=0)

        if schema_file_path:
            with open(schema_file_path, "r") as schema_file:
                schema = json.load(schema_file)
        else:
            schema = {}

        results = processor.generate_output(pdf_path=pdf_file_path, schema=schema)
        processor.save_output(results, output_file_path)
        print(f"Resultado guardado en: {output_file_path}")

    except Exception as e:
        print(f"Error: {e}")