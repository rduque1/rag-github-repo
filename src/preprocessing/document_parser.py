"""
Document parser using docling to extract text from various file formats.
"""

from pathlib import Path
import traceback

from docling.document_converter import DocumentConverter


def _parse_tsv(file_path: Path) -> str:
    """Parse a TSV file and convert to markdown table."""
    import pandas as pd
    df = pd.read_csv(file_path, sep='\t')
    return df.to_markdown(index=False)


def parse_document(file_path: str | Path) -> str:
    """
    Parse a document and extract its text content using docling.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, TSV, and more.

    :param file_path: Path to the document to parse.
    :return: Extracted text content from the document.
    """
    path = Path(file_path)

    # Handle TSV files separately (not supported by docling)
    if path.suffix.lower() == '.tsv':
        return _parse_tsv(path)

    try:
      converter = DocumentConverter()
      result = converter.convert(file_path)
      return result.document.export_to_markdown()
    except Exception as e:
      print(traceback.format_exc())


def parse_documents(file_paths: list[str | Path]) -> dict[str, str]:
    """
    Parse multiple documents and return their content.

    :param file_paths: List of paths to documents to parse.
    :return: Dictionary mapping file names to their extracted content.
    """
    converter = DocumentConverter()
    results = {}

    for file_path in file_paths:
        path = Path(file_path)
        # Handle TSV files separately
        if path.suffix.lower() == '.tsv':
            results[path.name] = _parse_tsv(path)
        else:
            result = converter.convert(file_path)
            results[path.name] = result.document.export_to_markdown()

    return results
