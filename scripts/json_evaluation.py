import os
import json
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from pathlib import Path

def load_json_pairs(base_dir):
    """
    Busca archivos JSON GT y los correspondientes de cada método de extracción en sus respectivas carpetas.
    """
    gt_dir = base_dir / "json_GT"
    methods = ["gpt", "gpt_schema", "gemini", "gemini_schema", "llama", "llama_schema"]

    file_pairs = []

    for gt_file in gt_dir.rglob("*_GT.json"):
        base_name = gt_file.stem.replace("_GT", "")
        for method in methods:
            method_file = base_dir / f"json_{method}" / f"{base_name}_{method}.json"

            # Validar la existencia del archivo
            if method_file.exists():
                file_pairs.append({
                    'GT': gt_file,
                    'method': method,
                    'extracted': method_file
                })
            else:
                print(f"Falta el archivo para {method_file}")

    return file_pairs


def levenshtein_similarity(str1, str2):
    """Calcula la similitud entre dos strings utilizando la distancia de Levenshtein (basado en SequenceMatcher)."""
    return SequenceMatcher(None, str1, str2).ratio()

def compare_json(gt_json, pred_json, path=""):
    results = {
        "correct": [],
        "missing": [],
        "extra": [],
        "mismatch": [],
        "similarity_scores": []
    }
    
    gt_keys = set(gt_json.keys())
    pred_keys = set(pred_json.keys())
    
    # Claves presentes en ambos JSON
    common_keys = gt_keys.intersection(pred_keys)
    
    # Claves que faltan o sobran
    results["missing"].extend([f"{path}.{key}" for key in gt_keys - pred_keys])
    results["extra"].extend([f"{path}.{key}" for key in pred_keys - gt_keys])
    
    for key in common_keys:
        gt_value = gt_json[key]
        pred_value = pred_json[key]
        current_path = f"{path}.{key}" if path else key
        
        # Si ambos valores son diccionarios, comparar recursivamente
        if isinstance(gt_value, dict) and isinstance(pred_value, dict):
            sub_results = compare_json(gt_value, pred_value, current_path)
            for k in results:
                results[k].extend(sub_results[k])
        
        # Si ambos valores son listas
        elif isinstance(gt_value, list) and isinstance(pred_value, list):
            # Si las listas contienen diccionarios, comparar de forma independiente del orden
            if all(isinstance(item, dict) for item in gt_value) and all(isinstance(item, dict) for item in pred_value):
                gt_list = sorted(gt_value, key=lambda x: sorted(x.items()))
                pred_list = sorted(pred_value, key=lambda x: sorted(x.items()))
                
                for i, (gt_item, pred_item) in enumerate(zip(gt_list, pred_list)):
                    sub_results = compare_json(gt_item, pred_item, f"{current_path}[{i}]")
                    for k in results:
                        results[k].extend(sub_results[k])
                
                # Verificar elementos adicionales o faltantes
                if len(gt_list) > len(pred_list):
                    extra_items = gt_list[len(pred_list):]
                    results["missing"].extend([f"{current_path}[{item}]" for item in extra_items])
                elif len(pred_list) > len(gt_list):
                    extra_items = pred_list[len(gt_list):]
                    results["extra"].extend([f"{current_path}[{item}]" for item in extra_items])
            
            # Si no son listas de diccionarios, comparar como conjuntos (independiente del orden)
            else:
                gt_set = set(map(str, gt_value))  # Convertir a cadenas para comparación
                pred_set = set(map(str, pred_value))
                
                correct = gt_set.intersection(pred_set)
                missing = gt_set - pred_set
                extra = pred_set - gt_set
                
                results["correct"].extend([f"{current_path}[{item}]" for item in correct])
                results["missing"].extend([f"{current_path}[{item}]" for item in missing])
                results["extra"].extend([f"{current_path}[{item}]" for item in extra])
        
        # Comparar valores simples usando Levenshtein similarity
        else:
            gt_str = str(gt_value)
            pred_str = str(pred_value)
            similarity = levenshtein_similarity(gt_str, pred_str)
            
            if gt_value == pred_value:
                results["correct"].append(current_path)
            else:
                results["mismatch"].append({
                    "path": current_path,
                    "expected": gt_value,
                    "predicted": pred_value
                })
            results["similarity_scores"].append({
                "path": current_path,
                "similarity": similarity,
                "expected": gt_str,
                "predicted": pred_str
            })
    
    return results


def calculate_metrics(results, gt_json, pred_json):
    """
    Calcula las métricas de evaluación basándose en los resultados de `compare_json`.
    
    Parámetros:
        results (dict): Diccionario generado por `compare_json` con claves:
                        - 'correct'
                        - 'missing'
                        - 'extra'
                        - 'mismatch'
                        - 'similarity_scores'

    Retorna:
        dict: Métricas calculadas, incluyendo:
              - accuracy
              - precision
              - recall
              - levenshtein_mean
    """
    # Extraer las métricas del resultado
    correct = len(results["correct"])
    missing = len(results["missing"])
    extra = len(results["extra"])
    mismatches = len(results["mismatch"])  # No se usa directamente en las métricas estándar

    total_keys = correct + missing + extra
    total_relevant = correct + missing  # Claves relevantes esperadas

    # Métricas básicas
    accuracy = correct / total_keys if total_keys > 0 else 0
    precision = correct / (correct + extra) if (correct + extra) > 0 else 0
    recall = correct / total_relevant if total_relevant > 0 else 0

    # Calcular media de similitud de Levenshtein
    similarity_scores = results["similarity_scores"]
    total_similarity = sum(score["similarity"] for score in similarity_scores)
    num_comparisons = len(similarity_scores)
    levenshtein_mean = total_similarity / num_comparisons if num_comparisons > 0 else 0

    # Calcular similitud de Levenshtein entre los JSON convertidos en string
    plain_levenshtein = levenshtein_similarity(str(gt_json), str(pred_json))

    # Generar el resultado final
    metrics = {
        "plain_levenshtein": plain_levenshtein,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "levenshtein_mean": levenshtein_mean,
        "total_correct": correct,
        "missing_keys": missing,
        "extra_keys": extra,
        "total_mismatches": mismatches,
        "total_keys": total_keys,
        "total_similarity_comparisons": num_comparisons,
        "mismatch_details": results["mismatch"],
        "similarity_scores" : similarity_scores
    }
    return metrics

def evaluate_extraction(base_dir, output_csv, output_pickle):
    """
    Evalúa todos los archivos JSON en una carpeta y guarda los resultados.
    """
    file_pairs = load_json_pairs(base_dir)
    overall_results = []

    for file_set in file_pairs:
        print(f"Evaluando {file_set['extracted'].stem} con el método {file_set['method']}...")
        with open(file_set['GT'], 'r', encoding='utf-8') as gt_file, \
             open(file_set['extracted'], 'r', encoding='utf-8') as pred_file:

            gt_json = json.load(gt_file)
            pred_json = json.load(pred_file)

            results = compare_json(gt_json, pred_json)
            metrics = calculate_metrics(results, gt_json, pred_json)
            metrics['document'] = file_set['GT'].stem.replace("_GT", "")
            metrics['method'] = file_set['method']
            overall_results.append(metrics)

    # Crear un DataFrame con los resultados
    df_results = pd.DataFrame(overall_results)

    # Exportar a CSV y Pickle
    df_results.to_csv(output_csv, index=False)
    df_results.to_pickle(output_pickle)

    print(f"Resultados guardados en {output_csv} y {output_pickle}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent / "data" / "evaluation"
    output_csv = Path(__file__).parent.parent / "data" / "results" / "results.csv"
    output_pickle = Path(__file__).parent.parent / "data" / "results" / "results.pickle"

    evaluate_extraction(base_dir, output_csv, output_pickle)