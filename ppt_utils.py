"""
PPT/PPTX utilities: convert PowerPoint files to PDF using LibreOffice,
then reuse the existing PDF -> image pipeline.

This approach avoids fragile, non-rendering parsers and preserves slide visuals.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def is_ppt(file_path: str) -> bool:
    """Return True if the file is a PPT/PPTX (by extension)."""
    suffix = Path(file_path).suffix.lower()
    return suffix in {".ppt", ".pptx"}


def _find_soffice() -> str | None:
    """Locate the LibreOffice 'soffice' executable."""
    # 1) PATH
    for name in ("soffice", "soffice.exe"):
        path = shutil.which(name)
        if path:
            return path

    # 2) Common Windows install paths
    candidates = []
    program_files = os.environ.get("PROGRAMFILES")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
    if program_files:
        candidates.append(os.path.join(program_files, "LibreOffice", "program", "soffice.exe"))
    if program_files_x86:
        candidates.append(os.path.join(program_files_x86, "LibreOffice", "program", "soffice.exe"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def ppt_to_pdf(ppt_path: str, output_dir: str | None = None) -> str:
    """
    Convert a PPT/PPTX file to PDF using LibreOffice (soffice).

    Returns:
        Path to the converted PDF file.
    """
    ppt_path = os.path.abspath(ppt_path)
    if output_dir is None:
        stem = Path(ppt_path).stem
        output_dir = os.path.join(os.path.dirname(ppt_path), f"{stem}_converted")
    os.makedirs(output_dir, exist_ok=True)

    if sys.platform.startswith("win"):
        # Windows: use PowerPoint COM automation (requires Microsoft Office installed)
        try:
            import win32com.client  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pywin32 is required for PowerPoint COM automation. "
                "Install it with: pip install pywin32"
            ) from exc

        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        # PowerPoint does not allow hiding the app window via Visible=0.
        # Keep it visible but open the presentation windowless.
        powerpoint.Visible = True
        try:
            presentation = powerpoint.Presentations.Open(ppt_path, WithWindow=False)
            # 32 = ppSaveAsPDF
            pdf_path = os.path.join(output_dir, f"{Path(ppt_path).stem}.pdf")
            presentation.SaveAs(pdf_path, 32)
            presentation.Close()
        finally:
            powerpoint.Quit()
    else:
        # Linux/macOS: use LibreOffice
        soffice = _find_soffice()
        if not soffice:
            raise ImportError(
                "LibreOffice is required to convert PPT/PPTX to PDF. "
                "Install it and ensure 'soffice' is on PATH."
            )
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            ppt_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"LibreOffice conversion failed: {msg}")

    # LibreOffice outputs <stem>.pdf in the outdir.
    pdf_path = os.path.join(output_dir, f"{Path(ppt_path).stem}.pdf")
    if not os.path.exists(pdf_path):
        # Fallback: pick any pdf in the outdir
        for candidate in Path(output_dir).glob("*.pdf"):
            pdf_path = str(candidate)
            break

    if not os.path.exists(pdf_path):
        raise RuntimeError("PDF conversion succeeded but no PDF was produced.")

    return pdf_path
