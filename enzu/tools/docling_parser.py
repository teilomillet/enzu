"""
Docling-based document parser for PDF and document processing.

Provides document parsing capabilities for the RLM context, allowing models to:
1. Parse PDFs and other documents into clean markdown text
2. Cache parsed documents by path + modification time
3. Format multiple documents for injection into RLM context

Usage:
    from enzu import Enzu

    # Single query with documents
    client = Enzu()
    result = client.run("Summarize the findings", documents=["paper.pdf"], tokens=2000)

    # Multi-turn session with cached documents
    session = client.session(documents=["paper.pdf", "appendix.pdf"])
    session.run("What is the methodology?")
    session.run("What are the results?")

Installation:
    pip install enzu[docling]
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Lazy import check for docling
_HAS_DOCLING = False
_DocumentConverter: Any = None
try:
    from docling.document_converter import DocumentConverter as _DocumentConverter  # type: ignore[import-not-found]

    _HAS_DOCLING = True
except ImportError:
    pass


class DoclingNotAvailable(ImportError):
    """Raised when docling is not installed but document parsing is requested."""

    def __init__(self) -> None:
        super().__init__(
            "Docling is required for document parsing but not installed.\n"
            "Install with: pip install enzu[docling]"
        )


@dataclass
class ParsedDocument:
    """Result of parsing a single document."""

    path: str
    filename: str
    content: str
    error: Optional[str] = None
    page_count: int = 0
    char_count: int = 0

    @property
    def success(self) -> bool:
        """True if document was parsed successfully."""
        return self.error is None and bool(self.content)


@dataclass
class _CacheEntry:
    """Internal cache entry with modification time."""

    mtime: float
    parsed: ParsedDocument


class DocumentCache:
    """
    Thread-safe document cache keyed by path + modification time.

    Documents are re-parsed only when the file is modified.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._converter: Any = None

    def _get_converter(self) -> Any:
        """Get or create the Docling converter (lazy initialization)."""
        if self._converter is None:
            if not _HAS_DOCLING or _DocumentConverter is None:
                raise DoclingNotAvailable()
            self._converter = _DocumentConverter()
        return self._converter

    def get(self, path: str) -> Optional[ParsedDocument]:
        """
        Get cached document if still valid (same mtime).

        Returns None if not in cache or file was modified.
        """
        abs_path = os.path.abspath(path)
        try:
            current_mtime = os.path.getmtime(abs_path)
        except OSError:
            return None

        with self._lock:
            entry = self._cache.get(abs_path)
            if entry is not None and entry.mtime == current_mtime:
                return entry.parsed
        return None

    def put(self, path: str, parsed: ParsedDocument, mtime: float) -> None:
        """Store parsed document in cache."""
        abs_path = os.path.abspath(path)
        with self._lock:
            self._cache[abs_path] = _CacheEntry(mtime=mtime, parsed=parsed)

    def clear(self) -> None:
        """Clear all cached documents."""
        with self._lock:
            self._cache.clear()

    def parse(self, path: str) -> ParsedDocument:
        """
        Parse a document, using cache if available.

        Args:
            path: Path to the document file (PDF, DOCX, etc.)

        Returns:
            ParsedDocument with content or error information.

        Raises:
            FileNotFoundError: If the file does not exist.
            DoclingNotAvailable: If docling is not installed.
        """
        abs_path = os.path.abspath(path)
        filename = os.path.basename(abs_path)

        # Check file exists
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Document not found: {abs_path}")

        # Get current modification time
        try:
            current_mtime = os.path.getmtime(abs_path)
        except OSError as e:
            return ParsedDocument(
                path=abs_path,
                filename=filename,
                content="",
                error=f"Cannot access file: {e}",
            )

        # Check cache
        cached = self.get(abs_path)
        if cached is not None:
            return cached

        # Parse with Docling
        try:
            converter = self._get_converter()
            result = converter.convert(abs_path)

            # Export to markdown
            content = result.document.export_to_markdown()
            page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0

            parsed = ParsedDocument(
                path=abs_path,
                filename=filename,
                content=content,
                page_count=page_count,
                char_count=len(content),
            )
        except DoclingNotAvailable:
            raise
        except Exception as e:
            parsed = ParsedDocument(
                path=abs_path,
                filename=filename,
                content="",
                error=f"Parse error: {e}",
            )

        # Cache result
        self.put(abs_path, parsed, current_mtime)
        return parsed


# Global cache instance
_document_cache = DocumentCache()


def documents_available() -> bool:
    """Check if Docling is installed and available."""
    return _HAS_DOCLING


def parse_document(path: Union[str, Path]) -> ParsedDocument:
    """
    Parse a single document using Docling.

    Uses caching based on file path and modification time.

    Args:
        path: Path to the document (PDF, DOCX, etc.)

    Returns:
        ParsedDocument with content or error.

    Raises:
        FileNotFoundError: If file does not exist.
        DoclingNotAvailable: If docling is not installed.

    Example:
        doc = parse_document("paper.pdf")
        if doc.success:
            print(f"Parsed {doc.char_count} characters from {doc.filename}")
        else:
            print(f"Error: {doc.error}")
    """
    return _document_cache.parse(str(path))


def parse_documents(paths: List[Union[str, Path]]) -> str:
    """
    Parse multiple documents and format for RLM context injection.

    Args:
        paths: List of document paths to parse.

    Returns:
        Formatted string with all document contents, suitable for
        injection into RLM context.

    Raises:
        DoclingNotAvailable: If docling is not installed.

    Example:
        context = parse_documents(["paper.pdf", "appendix.pdf"])
        # Returns formatted text with clear document boundaries
    """
    if not paths:
        return ""

    if not documents_available():
        raise DoclingNotAvailable()

    sections = []
    for path in paths:
        try:
            doc = parse_document(path)
            if doc.success:
                sections.append(
                    f"== Document: {doc.filename} ==\n"
                    f"{doc.content}\n"
                    f"== End: {doc.filename} =="
                )
            else:
                sections.append(
                    f"== Document: {doc.filename} ==\n"
                    f"[Error parsing document: {doc.error}]\n"
                    f"== End: {doc.filename} =="
                )
        except FileNotFoundError as e:
            filename = os.path.basename(str(path))
            sections.append(
                f"== Document: {filename} ==\n"
                f"[File not found: {e}]\n"
                f"== End: {filename} =="
            )

    return "\n\n".join(sections)


def get_cache() -> DocumentCache:
    """Get the global document cache instance."""
    return _document_cache


def clear_cache() -> None:
    """Clear the global document cache."""
    _document_cache.clear()


# Export for tools/__init__.py integration
DOCLING_TOOLS = {
    "parse_document": parse_document,
    "parse_documents": parse_documents,
    "documents_available": documents_available,
    "clear_cache": clear_cache,
}
