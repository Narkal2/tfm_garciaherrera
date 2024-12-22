import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from services.backend.src.modules.info_extraction import PDFProcessor

def process_pdf_batch(pdf_file: Path, methods: list, schema: Dict[str, Any]):
    """
    Procesa un archivo PDF y genera salidas en las combinaciones de métodos especificadas.

    Args:
        pdf_file (Path): Ruta del archivo PDF.
        methods (list): Lista de métodos de inferencia a usar.
        schema (dict): Esquema JSON a utilizar en los métodos que lo requieran.
    """
    for method in methods:
        llm_type = method.split("_")[0]
        use_schema = "schema" in method

        output_dir = Path(__file__).parent.parent / "data" / "evaluation" / f"json_{method}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pdf_file.stem}_{method}.json"

        if output_file.exists():
            print(f"Archivo ya existe: {output_file}")

        else:
            if llm_type == "llama": llm_type = "Llama-3.3" 
            elif llm_type == "gpt": llm_type = "GPT-4o-mini" 
            elif llm_type == "gemini": llm_type = "Gemini-1.5-flash" 
            processor = PDFProcessor(llm_type=llm_type)
            processor.init_llm(temperature=0)

            schema_to_use = schema if use_schema else {}

            try:
                results = processor.generate_output(pdf_path=str(pdf_file), schema=schema_to_use)
                processor.save_output(results, output_file)
                print(f"Guardado: {output_file}")
            except Exception as e:
                print(f"Error procesando {pdf_file} con {method}: {e}")


if __name__ == "__main__":
    """
    Eso de la clase PDFProcessor para extracción en batch de los json.
    """

    load_dotenv()

    base_dir = Path(__file__).parent.parent / "data" / "pdf_fitosanitario"
    methods = ["gpt", "gpt_schema", "gemini", "gemini_schema", "llama", "llama_schema"]
    schema_path = Path(__file__).parent.parent / "data" / "fitosanitario_schema.json"

    schema = {}
    if schema_path.exists():
        with open(schema_path, "r") as schema_file:
            print(f"Usando esquema: {schema_path}")
            schema = json.load(schema_file)

    for pdf_file in base_dir.glob("*.pdf"):
        print(pdf_file)
        process_pdf_batch(pdf_file, methods, schema)