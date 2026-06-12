"""LLM-tunner: a desktop GUI to customize open LLMs with PDF knowledge.

Two complementary paths are offered:

* **RAG** (default) — index PDF content into a local vector store and retrieve it at
  query time. This is the reliable way to *inject document knowledge* and works on
  CPU, Apple Silicon (MPS) and CUDA.
* **Fine-tuning** (optional) — QLoRA/LoRA training on Q&A pairs synthesised from the
  PDFs. This adapts the model's *style/format/behaviour* and requires a GPU for the
  4-bit (bitsandbytes) path.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
