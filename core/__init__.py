"""
BLUE-Gen Core - Modules IA Multimodal pour Assistant Eau Intelligent

Trois modules spécialisés:
  1. text_rag: Pipeline RAG pour analyse de documents (Retrieval-Augmented Generation)
  2. image_lora: Fine-tuning LoRA pour génération d'images spécialisées
  3. signal_forecast: Forecasting pour prédiction de séries temporelles
"""

from .text_rag import TextRAGPipeline, RAGPipeline
from .image_lora import ImageLoRAPipeline, LoRAPipeline
from .signal_forecast import SignalForecastPipeline, ForecastPipeline
from .utils import (
    get_env_vars,
    setup_chroma_client,
    create_collection,
    chunk_text,
    format_citations,
    load_or_create_embedding_model
)

__version__ = "1.0.0"
__author__ = "BLUE-Gen Team"

__all__ = [
    # Modules
    "TextRAGPipeline",
    "RAGPipeline",
    "ImageLoRAPipeline",
    "LoRAPipeline",
    "SignalForecastPipeline",
    "ForecastPipeline",
    # Utils
    "get_env_vars",
    "setup_chroma_client",
    "create_collection",
    "chunk_text",
    "format_citations",
    "load_or_create_embedding_model"
]
