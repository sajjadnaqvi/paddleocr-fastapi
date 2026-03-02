"""
Compatibility shim for paddlex (bundled inside paddleocr) which imports from
old LangChain 0.1/0.2 module paths that were removed in LangChain 0.3+.

Missing paths and their modern equivalents:
  langchain.docstore.document.Document      -> langchain_core.documents.Document
  langchain.text_splitter.*                 -> langchain_text_splitters.*

We inject fake modules into sys.modules so the imports succeed without
downgrading LangChain.

Also disables PaddlePaddle's OneDNN (MKL-DNN) backend which has a known
bug on Windows with PaddlePaddle 3.x:
  "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"

Import this module BEFORE importing paddleocr or paddlex anywhere.
"""
import os
import sys
from types import ModuleType

# ── Disable PaddlePaddle OneDNN / MKL-DNN backend ────────────────────────────
# Must be set before paddle / paddlex is imported. Fixes:
#   ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<...>]
#   (paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc)
#
# PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT  — PaddleX flag (read at paddlex import)
#   prevents PaddleX choosing mkldnn run_mode by default on CPU.
# FLAGS_use_mkldnn — PaddlePaddle C++ gflag, belt-and-suspenders fallback.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")


def _patch_langchain_compat() -> None:
    """Inject backward-compat shims for removed langchain module paths."""

    # ── langchain.docstore.document ──────────────────────────────────────────
    if "langchain.docstore.document" not in sys.modules:
        try:
            from langchain_core.documents import Document
        except ImportError:
            Document = None  # type: ignore[assignment,misc]

        if Document is not None:
            docstore_mod = ModuleType("langchain.docstore")
            document_mod = ModuleType("langchain.docstore.document")
            document_mod.Document = Document  # type: ignore[attr-defined]
            sys.modules.setdefault("langchain.docstore", docstore_mod)
            sys.modules["langchain.docstore.document"] = document_mod

    # ── langchain.text_splitter ──────────────────────────────────────────────
    if "langchain.text_splitter" not in sys.modules:
        try:
            import langchain_text_splitters as _lts
        except ImportError:
            _lts = None  # type: ignore[assignment]

        if _lts is not None:
            # Create a shim that re-exports everything from langchain_text_splitters
            text_splitter_mod = ModuleType("langchain.text_splitter")
            text_splitter_mod.__dict__.update(
                {k: v for k, v in _lts.__dict__.items() if not k.startswith("__")}
            )
            sys.modules["langchain.text_splitter"] = text_splitter_mod


_patch_langchain_compat()
