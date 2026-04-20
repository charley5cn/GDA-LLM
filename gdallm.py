from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import json
import random
import csv
import re
import sys
import resource
import logging
import subprocess
import psycopg2
import psutil
import time
import openai
import os

from src.rag.reranker import Reranker
from src.rag.adaptive_retrieval import AdaptiveRetriever
from src.rag.mmr_diversity import MMRDiversifier

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Task definitions
TASKS = {
    # "sentiment": {
    #     "name": "Sentiment Analysis",
    #     "prompt_template": (
    #         "You are an expert in sentiment analysis. "
    #         "Each review should be classified into one of three categories based on the sentiment it expresses: "
    #         "negative, neutral, or positive. "
    #         "Carefully read the review, analyze the tone and language used, and respond with only one of the following labels: "
    #         "negative, neutral, or positive, without any additional words.\n"
    #         f"Review to be classified: {{text}}."
    #     ),
    #     "output_format": "text"
    # }
    # "medical_extraction": {
    #     "name": "Medical Information Extraction",
    #     "prompt_template": (
    #         "You are an expert in medical information extraction. "
    #         "Given a medication description, extract the following fields: 'drug' and 'brand'. "
    #         "Use exactly the same text as it appears in the description, without translating, modifying, or correcting. "
    #         "If any of the fields are missing or not mentioned, return 'unknown' for that field. "
    #         "Return the result strictly in JSON format only, without any explanations or extra text.\n\n"
    #         "Example:\n"
    #         "Medication description: AC VALPROICO 500MG COMPRIMIDOS BIOLAB\n"
    #         "Expected output:\n"
    #         "{\"drug\": \"AC VALPROICO\", \"brand\": \"BIOLAB\"}\n\n"
    #         f"Medication description: {{text}}\n"
    #     ),
    #     "output_format": "json"
    # },
    # "review_classification": {
    #     "name": "Review Classification",
    #     "prompt_template": (
    #         "Você é um especialista em classificar avaliações. "
    #         "Essas avaliações podem ser enquadradas em quatro categorias: "
    #         "Delivery: problemas com entrega ou entregadores. "
    #         "Quality: questões sobre sabor, ingredientes, forma de preparo etc. "
    #         "Quantity: quantidade de alimentos ou itens faltantes. "
    #         "Praise: elogios ou sugestões positivas. "
    #         "Analise cuidadosamente o conteúdo e responda apenas com uma das categorias: Delivery, Quality, Quantity ou Praise, "
    #         "sem nenhuma palavra adicional.\n"
    #         f"Avaliação a ser classificada: {{text}}."
    #     ),
    #     "output_format": "text"
    # },
    # "machine_extraction": {
    #     "name": "Industrial Machinery Information Extraction",
    #     "prompt_template": (
    #         "You are an expert in industrial machinery information extraction. "
    #         "Given a machine description, extract the following fields: 'brand' and 'model'. "
    #         "Use exactly the same text as it appears in the description, without translating, modifying, or correcting. "
    #         "If any of the fields are missing or not mentioned, return 'unknown' for that field. "
    #         "Return the result strictly in JSON format only, without any explanations or extra text.\n\n"
    #         "Example:\n"
    #         "Machine description: John Deere 5075E Utility Tractor\n"
    #         "Expected output:\n"
    #         "{\"brand\": \"John Deere\", \"model\": \"5075E\"}\n\n"
    #         f"Machine description: {{text}}\n"
    #     ),
    #     "output_format": "json"
    # },
    "fake_news": {
        "name": "Fake News Detection",
        "prompt_template": (
            "Classify the following news as 'real' or 'fake'. "
            "A 'fake' news article may contain false, exaggerated, or misleading information, "
            "often displaying a sensationalist tone, lack of logical support, or internal inconsistencies. "
            "A 'real' news article is coherent, plausible, and free of evident contradictions. "
            "Carefully analyze the content and respond only with 'real' or 'fake', without any additional words. "
            f"News to be classified: {{text}}."
        ),
        "output_format": "text"
    }
}

# Feedback prompts for different tasks
FEEDBACK_PROMPTS = {
    # "sentiment": (
    #     "You are wrong. The correct sentiment is '{correct_answer}'. "
    #     "Please respond with only the correct sentiment, without any additional words."
    # )
    # "medical_extraction": (
    #     "The correct extraction should be: {correct_answer}. "
    #     "Please provide the correct JSON output without any additional text."
    # ),
    # "review_classification": (
    #     "A classificação correta é '{correct_answer}'. "
    #     "Por favor, responda apenas com a categoria correta, sem palavras adicionais."
    # ),
    # "machine_extraction": (
    #     "The correct extraction should be: {correct_answer}. "
    #     "Please provide the correct JSON output without any additional text."
    # ),
    "fake_news": (
        "The correct classification is '{correct_answer}'. "
        "Please respond with only 'real' or 'fake', without any additional words."
    )
}

# Check if tiktoken is available for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Initialize Hugging Face embedding model (use local path)
hf_embedding_model = SentenceTransformer(
    "./models/all-MiniLM-L6-v2")

# ------------------------- Helper Functions -------------------------
def approximate_token_count(text: str) -> int:
    """A simple approximation of token count using whitespace splitting."""
    return len(text.split())


def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken if available,
    else approximates via whitespace splitting.
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logging.warning("Error counting tokens with tiktoken: " + str(e))
            return len(text.split())
    else:
        return len(text.split())


def separate_thought_and_final(text: str) -> Tuple[str, str]:
    """
    Separates the chain of thought from the final answer if the text
    contains the tags <think> and </think>. Otherwise, returns an empty
    chain and the entire text.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        chain_of_thought = match.group(1).strip()
        final_answer = re.sub(
            pattern, "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
    else:
        chain_of_thought = ""
        final_answer = text.strip()
    return chain_of_thought, final_answer


def initialize_client(api_key: str) -> openai.OpenAI:
    """
    Initializes the OpenAI client with the provided API key.
    """
    openai.api_key = api_key
    return openai


def get_connection_string() -> str:
    """
    据环境变量构建 PostgreSQL 连接字符串
    """
    dbname = os.getenv("POSTGRES_DBNAME")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    return f"dbname='{dbname}' user='{user}' password='{password}' host='{host}' port='{port}'"


def load_reviews_dataset_from_postgres(
    conn_str: str,
    schema_name: str,
    table_name: str,
    review_column: str,
    label_column: Optional[str] = None
) -> List[Dict]:
    """
    从指定的模式（schema）和表（table）中加载数据
    要求包含以下列：id、review_column（例如：title），以及可选的 label_column（用于分类）
    返回一个字典列表
    """
    reviews = []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        if label_column:
            query = f"SELECT id, {review_column}, {label_column} FROM {schema_name}.{table_name};"
        else:
            query = f"SELECT id, {review_column} FROM {schema_name}.{table_name};"
        cursor.execute(query)
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            dataset_id = row[0] if row[0] is not None else i + 1
            if label_column:
                reviews.append({
                    "dataset_id": dataset_id,
                    "review": row[1],
                    "label": row[2],
                    "source_schema": schema_name
                })
            else:
                reviews.append({
                    "dataset_id": dataset_id,
                    "review": row[1],
                    "label": None,
                    "source_schema": schema_name
                })
        cursor.close()
        conn.close()
        logging.info(
            f"{len(reviews)} records loaded from {schema_name}.{table_name}."
        )
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
    return reviews

# ------------------------- Classification Functions -------------------------
def classify_review(
    client: Optional[openai.OpenAI],
    review: str,
    provider: str,
    model_name: str,
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, Optional[float], Optional[float],
           Optional[int], Optional[int], Optional[int]]:
    """
    Routes the classification request to the appropriate provider.
    """
    if provider.lower() == "ollama":
        return ollama_classify(model_name, review, additional_context, task)
    elif provider.lower() == "gemini":
        return gemini_classify(model_name, review, additional_context, task)
    elif provider.lower() == "openrouter":
        return openrouter_classify(client, review, model_name, additional_context, task)
    else:
        return openai_classify(client, review, model_name, additional_context, task)


def openai_classify(
    client: openai.OpenAI,
    review: str,
    model_name: str = "gpt-4o-mini",
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, float, float, Optional[int], Optional[int], Optional[int]]:
    """
    Uses the OpenAI API to classify the text according to the specified task.
    """
    prompt = (additional_context + "\n") if additional_context else ""
    task_template = TASKS[task]["prompt_template"]
    prompt += task_template.format(text=review)

    start_time = time.perf_counter()
    cpu_before = time.process_time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024**2)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.perf_counter()
    cpu_after = time.process_time()
    mem_after = proc.memory_info().rss / (1024**2)
    latency = end_time - start_time
    cpu_time = cpu_after - cpu_before
    mem_usage = mem_after - mem_before
    llm_output = completion.choices[0].message.content.strip()

    # For JSON output format, try to parse and validate the JSON
    if TASKS[task]["output_format"] == "json":
        try:
            json.loads(llm_output)
        except json.JSONDecodeError:
            logging.warning(
                f"Invalid JSON output for task {task}: {llm_output}")
            llm_output = "{}"

    try:
        usage = completion["usage"]
    except (TypeError, KeyError):
        usage = {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # 如果无法获取 OpenAI 的 usage 信息，则尝试使用 tiktoken 作为备用方案
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(encoding.encode(llm_output))
                total_tokens = prompt_tokens + completion_tokens
                logging.info(
                    "Tokens calculated via tiktoken as fallback for OpenAI."
                )
            except Exception as e:
                logging.warning(
                    "Error calculating tokens via tiktoken: " + str(e)
                )
                prompt_tokens, completion_tokens, total_tokens = None, None, None

    return (
        llm_output,
        latency,
        cpu_time,
        mem_usage,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def ollama_classify(
    model_name: str,
    review: str,
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, Optional[float], Optional[float],
           Optional[int], Optional[int], Optional[int]]:
    """
    Uses a subprocess call to the Ollama model for classification.
    """
    prompt = (additional_context + "\n") if additional_context else ""
    task_template = TASKS[task]["prompt_template"]
    prompt += task_template.format(text=review)

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt, timeout=60)
        process.wait(timeout=10)
    except Exception as e:
        logging.error(f"Error calling Ollama model: {e}")
        return "", 0.0, None, None, None, None, None

    end_time = time.perf_counter()
    latency = end_time - start_time

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_time = usage.ru_utime + usage.ru_stime
    mem_usage = (
        (usage.ru_maxrss / (1024 * 1024))
        if sys.platform == "darwin"
        else usage.ru_maxrss / 1024.0
    )

    llm_output = output.strip() if output else ""

    # For JSON output format, try to parse and validate the JSON
    if TASKS[task]["output_format"] == "json":
        try:
            json.loads(llm_output)
        except json.JSONDecodeError:
            logging.warning(
                f"Invalid JSON output for task {task}: {llm_output}")
            llm_output = "{}"

    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(llm_output))
            total_tokens = prompt_tokens + completion_tokens
            logging.info("Tokens calculated via tiktoken for Ollama.")
        except Exception as e:
            logging.warning(
                "Error calculating tokens via tiktoken for Ollama: " + str(e)
            )
            prompt_tokens, completion_tokens, total_tokens = None, None, None
    else:
        prompt_tokens, completion_tokens, total_tokens = None, None, None

    return (
        llm_output,
        latency,
        cpu_time,
        mem_usage,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


# def gemini_classify(
#     model_name: str,
#     review: str,
#     additional_context: str = "",
#     task: str = "review_classification"
# ) -> Tuple[str, float, Optional[float], Optional[float],
#            Optional[int], Optional[int], Optional[int]]:
#     """
#     Uses the Gemini API to classify the text according to the specified task.
#     """
#     prompt = (additional_context + "\n") if additional_context else ""
#     task_template = TASKS[task]["prompt_template"]
#     prompt += task_template.format(text=review)

#     start_time = time.perf_counter()
#     cpu_before = time.process_time()
#     proc = psutil.Process()
#     mem_before = proc.memory_info().rss / (1024**2)

#     from google import genai
#     gemini_client = genai.Client(api_key=os.getenv("API_KEY_GOOGLE"))
#     response = gemini_client.models.generate_content(
#         model=model_name,
#         contents=prompt
#     )

#     end_time = time.perf_counter()
#     cpu_after = time.process_time()
#     mem_after = proc.memory_info().rss / (1024**2)

#     latency = end_time - start_time
#     cpu_time = cpu_after - cpu_before
#     mem_usage = mem_after - mem_before

#     llm_output = response.text.strip() if response.text else ""

#     # For JSON output format, try to parse and validate the JSON
#     if TASKS[task]["output_format"] == "json":
#         try:
#             json.loads(llm_output)
#         except json.JSONDecodeError:
#             logging.warning(
#                 f"Invalid JSON output for task {task}: {llm_output}")
#             llm_output = "{}"

#     if TIKTOKEN_AVAILABLE:
#         try:
#             encoding = tiktoken.get_encoding("cl100k_base")
#             prompt_tokens = len(encoding.encode(prompt))
#             completion_tokens = len(encoding.encode(llm_output))
#             total_tokens = prompt_tokens + completion_tokens
#             logging.info("Tokens calculated via tiktoken for Gemini.")
#         except Exception as e:
#             logging.warning(
#                 "Error calculating tokens via tiktoken for Gemini: " + str(e)
#             )
#             prompt_tokens, completion_tokens, total_tokens = None, None, None
#     else:
#         prompt_tokens, completion_tokens, total_tokens = None, None, None

#     return (
#         llm_output,
#         latency,
#         cpu_time,
#         mem_usage,
#         prompt_tokens,
#         completion_tokens,
#         total_tokens,
#     )


def openrouter_classify(
    client: Any,
    review: str,
    model_name: str,
    additional_context: Optional[str] = None,
    task: Optional[str] = None
) -> Tuple[str, float, float, float, int, int, int]:
    """
    Uses the OpenRouter API (via the OpenAI-compatible interface)
    to classify a review based on the specified task.
    """
    start_time = time.perf_counter()
    cpu_before = time.process_time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024**2)

    try:
        # Construct the prompt based on the task
        task_template = TASKS[task]["prompt_template"]
        prompt = task_template.format(text=review)
        if additional_context:
            prompt = f"{additional_context}\n{prompt}"

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        end_time = time.perf_counter()
        cpu_after = time.process_time()
        mem_after = proc.memory_info().rss / (1024**2)

        latency = end_time - start_time
        cpu_time = cpu_after - cpu_before
        mem_usage = mem_after - mem_before

        llm_output = response.choices[0].message.content.strip()

        # For JSON output format, try to parse and validate the JSON
        if TASKS[task]["output_format"] == "json":
            try:
                json.loads(llm_output)
            except json.JSONDecodeError:
                logging.warning(
                    f"Invalid JSON output for task {task}: {llm_output}")
                llm_output = "{}"

        return (
            llm_output,
            latency,
            cpu_time,
            mem_usage,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens
        )

    except Exception as e:
        logging.error(f"Error calling OpenRouter API: {e}")
        return ("error", 0.0, 0.0, 0.0, 0, 0, 0)

# ------------------------- Graph-RAG Functions -------------------------
def get_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Generates an embedding for the provided text using the
    Hugging Face model (SentenceTransformer).
    """
    try:
        embedding = hf_embedding_model.encode(text).tolist()
        if not embedding:
            logging.error("Empty embedding generated by HF model.")
            return [0.0] * dim
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding with HF model: {e}")
        return [0.0] * dim


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算两个向量间的余弦相似度
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_context(
    review_text: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    top_n: int = 3
) -> List[Dict]:
    """
    增强版图检索方法：
    1. 将文本转换为嵌入向量。
    2. 查询 Neo4j 数据库中标签为：News 的节点（包含标题、分类、嵌入向量属性）。
    3. 计算文本嵌入向量与图中每个节点嵌入向量的余弦相似度。
    4. 选取相似度最高的 top_n 个节点。
    5. 返回包含完整文档信息的字典列表（用于后续优化模块）。
    """
    # 1. 将文本转换为嵌入向量
    text_embedding = get_embedding(review_text)

    # 2. 在 Neo4j 中仅查询标签为：News 的节点
    context_items = []
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result_news = session.run(
            "MATCH (n:News) RETURN n.title AS title, "
            "n.classification AS classification, n.embedding AS embedding"
        )
        for record in result_news:
            title = record["title"]
            classification = record["classification"]
            embedding = record["embedding"]
            if not embedding:
                continue
            # 3. 计算相似度
            sim = cosine_similarity(text_embedding, embedding)
            # 4. 存储为字典（包含所有信息）
            context_items.append({
                'title': title,
                'classification': classification,
                'embedding': embedding,
                'similarity': sim
            })

    driver.close()

    # 5. 按相似度排序并返回 top_n
    context_items.sort(key=lambda x: x['similarity'], reverse=True)
    return context_items[:top_n]

# ------------------------- Experiment and CSV Functions -------------------------


def export_records_to_csv(records: List[Dict], filename: str) -> None:
    """
    Exports records to a CSV file.
    """
    if not records:
        logging.warning("没有可导出的记录。")
        return
    headers = records[0].keys()
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
        logging.info(f"Records exported to CSV: {filename}")
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")


def create_or_update_experiment_table(
    conn_str: str,
    experiment_schema: str,
    table_name: str
) -> None:
    """
    Creates or updates the experiment table in PostgreSQL.
    """
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {experiment_schema}.{table_name} (
            experiment_id TEXT,
            dataset_id INTEGER NOT NULL,
            review TEXT,
            validation TEXT,
            llm_output TEXT,
            chain_of_thought TEXT,
            final_answer TEXT,
            is_correct BOOLEAN,
            latency FLOAT,
            cpu_time FLOAT,
            mem_usage FLOAT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            throughput FLOAT,
            energy_consumption FLOAT,
            flops_per_token FLOAT,
            classification_timestamp TIMESTAMP,
            experiment_date DATE,
            model_name TEXT,
            code_name TEXT,
            prompt TEXT,
            PRIMARY KEY (dataset_id, experiment_id)
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        alter_queries = [
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS chain_of_thought TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS final_answer TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS throughput FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS energy_consumption FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS flops_per_token FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS experiment_date DATE;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS model_name TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS code_name TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS prompt TEXT;"
        ]

        for query in alter_queries:
            cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(
            f"Experiment table {experiment_schema}.{table_name} is ready."
        )

    except Exception as e:
        logging.error(f"Error creating/updating experiment table: {e}")


def insert_experiment_records(
    conn_str: str,
    experiment_schema: str,
    table_name: str,
    records: List[Dict]
) -> None:
    """
    Inserts or updates a batch of records into the experiment table.
    """
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        insert_query = f"""
        INSERT INTO {experiment_schema}.{table_name} 
        (experiment_id, dataset_id, review, validation, llm_output, chain_of_thought,
         final_answer, is_correct, latency, cpu_time, mem_usage, 
         prompt_tokens, completion_tokens, total_tokens, throughput, 
         energy_consumption, flops_per_token, classification_timestamp, 
         experiment_date, model_name, code_name, prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_id, experiment_id) DO UPDATE SET
            review = EXCLUDED.review,
            validation = EXCLUDED.validation,
            llm_output = EXCLUDED.llm_output,
            chain_of_thought = EXCLUDED.chain_of_thought,
            final_answer = EXCLUDED.final_answer,
            is_correct = EXCLUDED.is_correct,
            latency = EXCLUDED.latency,
            cpu_time = EXCLUDED.cpu_time,
            mem_usage = EXCLUDED.mem_usage,
            prompt_tokens = EXCLUDED.prompt_tokens,
            completion_tokens = EXCLUDED.completion_tokens,
            total_tokens = EXCLUDED.total_tokens,
            throughput = EXCLUDED.throughput,
            energy_consumption = EXCLUDED.energy_consumption,
            flops_per_token = EXCLUDED.flops_per_token,
            classification_timestamp = EXCLUDED.classification_timestamp,
            experiment_date = EXCLUDED.experiment_date,
            model_name = EXCLUDED.model_name,
            code_name = EXCLUDED.code_name,
            prompt = EXCLUDED.prompt;
        """

        values = []
        for record in records:
            values.append((
                record["experiment_id"],
                record["dataset_id"],
                record["review"],
                record["validation"],
                record["llm_output"],
                record["chain_of_thought"],
                record["final_answer"],
                record["is_correct"],
                record["latency"],
                record["cpu_time"],
                record["mem_usage"],
                record["prompt_tokens"],
                record["completion_tokens"],
                record["total_tokens"],
                record["throughput"],
                record["energy_consumption"],
                record["flops_per_token"],
                record["classification_timestamp"],
                record["experiment_date"],
                record["model_name"],
                record["code_name"],
                record["prompt"]
            ))

        cursor.executemany(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(
            f"{len(records)} 条记录插入/更新到实验表中."
        )

    except Exception as e:
        logging.error(f"实验数据插入失败: {e}")


# ------------------------- Main Function -------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        # description=(
        #     "Processa experimentos de classificação via LLM com In-context learning e Graph-RAG. "
        #     "Para cada notícia, recupera contexto do grafo (via similaridade), "
        #     "constrói o prompt e classifica a notícia como 'real' ou 'fake'."
        # )
    )
    parser.add_argument("--source_schema", type=str,
                        required=True, help="源 schema（例如：test）")
    parser.add_argument("--table_name", type=str, required=True,
                        help="源表名称（例如：fake_news_gossipcop）")
    parser.add_argument("--review_column", type=str, required=True,
                        help="评论列名称（例如：title）")
    parser.add_argument("--label_column", type=str, required=False, default=None,
                        help="验证列名称（可选，例如：classification）")
    parser.add_argument("--model_name", type=str, required=True,
                        help="LLM 模型名称（例如：gpt-4o-mini、ollama、openrouter 或 gemini）")
    parser.add_argument("--code_name", type=str, required=False,
                        default="rag", help="代码名称 (ex.: rag)")
    parser.add_argument("--provider", type=str, required=False, default="openai",
                        choices=["openai", "ollama", "openrouter", "gemini"],
                        help="要使用的提供方（openai、ollama、openrouter 或 gemini）")
    parser.add_argument("--api_key", type=str, required=False,
                        help="提供者的 API 密钥（若未提供，将从 .env 文件中读取）")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASKS.keys()),
                        help="要执行的任务 (sentiment, medical_extraction, review_classification, machine_extraction, fake_news)")
    # Neo4j 认证信息
    parser.add_argument("--neo4j_uri", type=str,
                        default=os.getenv("NEO4J_URI"),
                        help="Neo4j 的 URI（若未提供，将从 .env 文件中读取）")
    parser.add_argument("--neo4j_user", type=str,
                        default=os.getenv("NEO4J_USER"),
                        help="Neo4j 用户名（若未提供，将从 .env 文件中读取）")
    parser.add_argument("--neo4j_password", type=str,
                        default=os.getenv("NEO4J_PASSWORD"),
                        help="Neo4j 密码（若未提供，将从 .env 文件中读取）")
    # 执行模式选择
    parser.add_argument("--execution_mode", type=str, default="rag",
                        choices=["base", "rag"],
                        help=(
                            "执行模式：'base'（无检索增强） "
                            "或 'rag'（增强RAG：自适应检索+MMR多样性+重排序）"
                        ))
    args = parser.parse_args()

    # 选择服务提供商
    if args.provider.lower() == "openai":
        api_key = args.api_key if args.api_key else os.getenv("API_KEY_OPENAI")
        if not api_key:
            logging.error(
                "未提供 API_KEY_OPENAI，或在 .env 文件中未找到 OpenAI 的密钥。")
            return
        openai.api_key = api_key
        client = openai  # agora client é o próprio módulo openai
    elif args.provider.lower() == "openrouter":
        if not os.getenv("OPENROUTER_TOKEN"):
            raise ValueError(
                "未提供 OPENROUTER_TOKEN，或在 .env 文件中未找到 OpenRouter 的令牌。")
        client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_TOKEN"), base_url="https://openrouter.ai/api/v1"
        )
    elif args.provider.lower() == "gemini":
        if not os.getenv("API_KEY_GOOGLE"):
            logging.error("未在 .env 文件中找到用于 Gemini 的 API_KEY_GOOGLE。")
            return
        client = None
    elif args.provider.lower() == "ollama":
        client = None
    else:
        client = None

    # 检查 Neo4j 凭据是否可用
    if not args.neo4j_uri or not args.neo4j_user or not args.neo4j_password:
        logging.error("未在 .env 文件中找到 Neo4j 的凭据。")
        return

    conn_str = get_connection_string()

    # 初始化增强RAG组件（仅在使用RAG或EcoRAG模式时）
    if args.execution_mode in ["rag", "ecorag"]:
        logging.info("初始化增强RAG组件...")

        # 自适应检索器：根据查询难度动态调整检索数量
        adaptive_retriever = AdaptiveRetriever()
        logging.info("  ✓ 自适应检索器已初始化")

        # MMR多样性优化器：保证检索结果的多样性
        mmr_diversifier = MMRDiversifier(lambda_param=0.7)
        logging.info("  ✓ MMR多样性优化器已初始化 (λ=0.7)")

        # 重排序器：使用语义模型提升结果相关性
        reranker = Reranker(use_cross_encoder=True)
        logging.info("  ✓ 重排序器已初始化")
    else:
        adaptive_retriever = None
        mmr_diversifier = None
        reranker = None
        logging.info("运行在 Base 模式，跳过增强RAG组件初始化")

    # 根据执行模式加载测试集和验证集数据
    test_dataset = load_reviews_dataset_from_postgres(
        conn_str, args.source_schema, args.table_name,
        args.review_column, args.label_column
    )

    reviews_dataset = test_dataset
    logging.info(f"总记录数: {len(reviews_dataset)}")

    # 创建或更新实验表
    experiment_schema = "experiments"
    create_or_update_experiment_table(
        conn_str, experiment_schema, args.table_name)

    batch_size = 2
    batch_records = []
    all_records = []

    # Estimativa de FLOPs do modelo (valor hipotético)
    model_flops_estimate = 1e11

    for i, review_data in enumerate(reviews_dataset, start=1):
        dataset_id = review_data["dataset_id"]
        review_text = review_data["review"]
        validation = review_data["label"]

        # 构建上下文（根据执行模式）
        if args.execution_mode == "base":
            combined_context = ""

        elif args.execution_mode == "rag":
            # RAG模式：使用增强检索流程（自适应→MMR→重排序）
            logging.info(f"[{i}/{len(reviews_dataset)}] 开始增强RAG检索...")

            # 步骤1：基础检索函数（包装）
            def base_retrieval(query, k):
                return retrieve_context(
                    query, args.neo4j_uri, args.neo4j_user,
                    args.neo4j_password, top_n=k
                )

            # 步骤2：轻量级探测评估难度（使用3个维度：语义不确定性+长度+特异性）
            difficulty, recall_k, difficulty_level, probe_results = adaptive_retriever.estimate_difficulty_with_lightweight_probe(
                query=review_text,
                retrieval_function=base_retrieval,
                probe_k=5  # 轻量级探测：只检索5个文档用于难度评估
            )
            logging.info(f"  难度评估: {difficulty:.3f} ({difficulty_level}), 目标召回: {recall_k}")

            # 步骤3：根据难度进行最终召回（探测结果仅用于评估，不作为最终候选）
            # 原因：需要给MMR足够的候选（至少20个）进行多样性优化
            candidates = base_retrieval(query=review_text, k=recall_k)
            logging.info(f"  最终召回: {len(candidates)}个候选")

            # 步骤4：MMR多样性优化（从召回中选择多样化的10个）
            if len(candidates) > 0:
                query_emb = get_embedding(review_text)
                mmr_k = min(10, len(candidates))  # 最多选10个
                candidates = mmr_diversifier.diversify(
                    query_embedding=query_emb,
                    candidates=candidates,
                    k=mmr_k,
                    embedding_field='embedding'
                )
                logging.info(f"  MMR优化: 从{len(candidates)}个候选中选出{mmr_k}个多样化结果")

            # 步骤5：Cross-Encoder重排序（精选Top 5）
            final_candidates = reranker.rerank(
                query=review_text,
                candidates=candidates,
                top_k=5,
                text_field='title'
            )
            logging.info(f"  重排序: 最终{len(final_candidates)}个结果")

            # 步骤6：构建上下文字符串
            combined_context = ""
            for candidate in final_candidates:
                combined_context += f"Example of {candidate['classification']}: {candidate['title']}\n"

        else:
            combined_context = ""

        # 构建初始提示（不含修正尝试）
        initial_prompt = (combined_context + "\n") if combined_context else ""
        task_template = TASKS[args.task]["prompt_template"]
        initial_prompt += task_template.format(text=review_text)

        # 首次调用分类
        (
            llm_output,
            latency,
            cpu_time,
            mem_usage,
            prompt_tokens,
            completion_tokens,
            total_tokens
        ) = classify_review(
            client,
            review_text,
            args.provider,
            args.model_name,
            combined_context,
            args.task
        )

        # 将思考过程 (<think> ... </think>) 与输出结果分离
        chain_of_thought, final_answer = separate_thought_and_final(llm_output)

        # 评估结果是否正确（如果有标签的话）
        if TASKS[args.task]["output_format"] == "json":
            try:
                # For JSON output, compare the parsed JSON with the validation
                if validation is not None:
                    try:
                        validation_json = json.loads(validation)
                        output_json = json.loads(final_answer)
                        is_correct = validation_json == output_json
                    except json.JSONDecodeError:
                        is_correct = False
                else:
                    is_correct = None
            except json.JSONDecodeError:
                is_correct = False
        else:
            # For text output, compare strings directly
            is_correct = (
                final_answer.lower() == validation.lower()
                if validation is not None
                else None
            )

        attempts = 1  # 重试计数器

        # 如果是来自 "validator" schema 的记录且存在标签,
        # 尝试通过附加提示来修正回答.
        if review_data.get("source_schema", "test").lower() == "validator" and validation is not None:
            while not is_correct and attempts <= 5:
                reattempt_message = FEEDBACK_PROMPTS[args.task].format(
                    correct_answer=validation
                )

                combined_context_retry = combined_context + "\n" + reattempt_message
                logging.info(
                    f"Attempt {attempts} para dataset_id {dataset_id}")

                # Nova tentativa
                (
                    retry_output,
                    retry_latency,
                    retry_cpu_time,
                    retry_mem_usage,
                    retry_prompt_tokens,
                    retry_completion_tokens,
                    retry_total_tokens
                ) = classify_review(
                    client,
                    review_text,
                    args.provider,
                    args.model_name,
                    combined_context_retry,
                    args.task
                )

                chain_of_thought, retry_final_answer = separate_thought_and_final(
                    retry_output)

                # 评估结果是否正确（如果存在标签的话）
                if TASKS[args.task]["output_format"] == "json":
                    try:
                        # For JSON output, compare the parsed JSON with the validation
                        validation_json = json.loads(validation)
                        output_json = json.loads(retry_final_answer)
                        is_correct = validation_json == output_json
                    except json.JSONDecodeError:
                        is_correct = False
                else:
                    # For text output, compare strings directly
                    is_correct = (retry_final_answer.lower()
                                  == validation.lower())

                final_answer = retry_final_answer
                llm_output = retry_output
                attempts += 1

                if is_correct:
                    logging.info(
                        f"在第 {attempts-1} 次尝试中获得了正确答案"
                        f"对于 dataset_id {dataset_id}"
                    )
                    break

                if attempts > 5:
                    logging.info(
                        f"经过 5 次尝试后，最终得到错误的回答 "
                        f"对于 dataset_id {dataset_id} 结果为 '{final_answer}'"
                    )
                    break

        classification_timestamp = datetime.now()
        experiment_date = classification_timestamp.date()
        experiment_id = f"{args.table_name}_{args.code_name}_{args.model_name}_{args.code_name}"

        # 吞吐量及其他指标
        throughput = (
            (total_tokens / latency)
            if (latency > 0 and total_tokens is not None)
            else None
        )
        energy_consumption = (
            (cpu_time * 50) if cpu_time is not None else None
        )
        flops_per_token = (
            (model_flops_estimate / total_tokens)
            if (total_tokens and total_tokens > 0)
            else None
        )

        record = {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "review": review_text,
            "validation": validation,
            "llm_output": llm_output,
            "chain_of_thought": chain_of_thought,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "latency": latency,
            "cpu_time": cpu_time,
            "mem_usage": mem_usage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "energy_consumption": energy_consumption,
            "flops_per_token": flops_per_token,
            "classification_timestamp": classification_timestamp,
            "experiment_date": experiment_date,
            "model_name": args.model_name,
            "code_name": args.code_name,
            "prompt": initial_prompt
        }

        batch_records.append(record)
        all_records.append(record)

        # 批量插入
        if i % batch_size == 0:
            insert_experiment_records(
                conn_str, experiment_schema, args.table_name, batch_records
            )
            logging.info(
                f"包含 {batch_size} 条记录的批次已插入数据库."
            )
            batch_records.clear()
            logging.info("等待 5 秒后再处理下一批次...")
            time.sleep(5)

    # 插入批次中剩余的记录
    if batch_records:
        insert_experiment_records(
            conn_str, experiment_schema, args.table_name, batch_records
        )
        logging.info(
            f"包含 {len(batch_records)} 条记录的最终批次已插入数据库."
        )

if __name__ == "__main__":
    main()

# python ecorag.py --source_schema test --table_name politifact --review_column title --label_column label --model_name llama3.2:1b --provider ollama --code_name ecorag --execution_mode ecorag --task sentiment