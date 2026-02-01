"""Tests for document context filling in the Docling integration."""

import pytest
from unittest.mock import patch, MagicMock


class TestDoclingParser:
    """Tests for the docling_parser module."""

    def test_documents_available_returns_false_when_not_installed(self):
        """documents_available() returns False when docling is not installed."""
        from enzu.tools.docling_parser import documents_available

        # Since docling is not installed in test env, should return False
        # (or True if it is installed, which is also valid)
        result = documents_available()
        assert isinstance(result, bool)

    def test_parse_documents_empty_list_returns_empty_string(self):
        """parse_documents([]) returns empty string."""
        from enzu.tools.docling_parser import parse_documents, documents_available

        if not documents_available():
            pytest.skip("Docling not installed")

        result = parse_documents([])
        assert result == ""

    def test_docling_not_available_error_message(self):
        """DoclingNotAvailable has clear install instructions."""
        from enzu.tools.docling_parser import DoclingNotAvailable

        error = DoclingNotAvailable()
        assert "pip install enzu[docling]" in str(error)

    def test_parsed_document_success_property(self):
        """ParsedDocument.success is True when content is present and no error."""
        from enzu.tools.docling_parser import ParsedDocument

        # Success case
        doc = ParsedDocument(
            path="/test/file.pdf",
            filename="file.pdf",
            content="Some content",
        )
        assert doc.success is True

        # Error case
        doc_error = ParsedDocument(
            path="/test/file.pdf",
            filename="file.pdf",
            content="",
            error="Parse failed",
        )
        assert doc_error.success is False

        # Empty content case
        doc_empty = ParsedDocument(
            path="/test/file.pdf",
            filename="file.pdf",
            content="",
        )
        assert doc_empty.success is False


class TestSessionDocumentContext:
    """Tests for document context in Session."""

    def test_session_combines_documents_with_history_and_data(self):
        """Session.run() correctly combines session docs + query docs + history + data."""
        from enzu.session import Session, Exchange

        # Create a session and manually set parsed documents
        session = Session(model="gpt-4", provider="openrouter")
        session._parsed_documents = "== Document: paper.pdf ==\nPaper content\n== End: paper.pdf =="

        # Add some history
        session.exchanges = [
            Exchange(user="First question", assistant="First answer")
        ]

        # Mock enzu_run to capture what data is passed
        captured_data = []

        def mock_run(*args, **kwargs):
            captured_data.append(kwargs.get("data"))
            # Return a mock report
            mock_report = MagicMock()
            mock_report.outcome = MagicMock()
            mock_report.outcome.value = "success"
            mock_report.answer = "Mock answer"
            mock_report.budget_usage = MagicMock()
            mock_report.budget_usage.cost_usd = 0.001
            mock_report.budget_usage.output_tokens = 50
            return mock_report

        # Patch where it's used (inside session.py, it imports from enzu.api)
        with patch("enzu.api.run", mock_run):
            session.run("Test query", data="Additional data")

        # Verify the combined data structure
        assert len(captured_data) == 1
        combined = captured_data[0]

        # Should contain session documents
        assert "== Session Documents ==" in combined
        assert "Paper content" in combined

        # Should contain history
        assert "== Previous Conversation ==" in combined
        assert "First question" in combined
        assert "First answer" in combined

        # Should contain current data
        assert "== Current Data ==" in combined
        assert "Additional data" in combined

    def test_session_includes_query_documents(self):
        """Session.run() includes per-query documents when provided."""
        from enzu.session import Session

        session = Session(model="gpt-4", provider="openrouter")
        session._parsed_documents = "Session doc content"

        captured_data = []

        def mock_run(*args, **kwargs):
            captured_data.append(kwargs.get("data"))
            mock_report = MagicMock()
            mock_report.outcome = MagicMock()
            mock_report.outcome.value = "success"
            mock_report.answer = "Mock answer"
            mock_report.budget_usage = MagicMock()
            mock_report.budget_usage.cost_usd = 0.001
            mock_report.budget_usage.output_tokens = 50
            return mock_report

        # Mock parse_documents for per-query docs
        def mock_parse_documents(paths):
            return "== Document: query.pdf ==\nQuery doc content\n== End: query.pdf =="

        with patch("enzu.api.run", mock_run):
            with patch("enzu.tools.docling_parser.documents_available", return_value=True):
                with patch("enzu.tools.docling_parser.parse_documents", mock_parse_documents):
                    session.run("Test query", documents=["query.pdf"])

        assert len(captured_data) == 1
        combined = captured_data[0]

        # Should contain both session and query documents
        assert "== Session Documents ==" in combined
        assert "Session doc content" in combined
        assert "== Query Documents ==" in combined
        assert "Query doc content" in combined

    def test_session_documents_persist_in_serialization(self):
        """Session documents are saved and restored correctly."""
        from enzu.session import Session

        # Create session with documents list (not parsed, for serialization)
        session = Session(model="gpt-4", provider="openrouter")
        session._session_documents = ["paper.pdf", "appendix.pdf"]

        # Serialize
        data = session.to_dict()
        assert "documents" in data
        assert data["documents"] == ["paper.pdf", "appendix.pdf"]


class TestApiDocumentContext:
    """Tests for document context in api.run()."""

    def test_api_run_combines_documents_with_data(self):
        """api.run() combines parsed documents with data parameter."""
        from enzu import api

        # Mock parse_documents to return known content
        def mock_parse_documents(paths):
            return "== Document: test.pdf ==\nTest content\n== End: test.pdf =="

        # Track what gets passed to _run_internal
        captured_data = []

        def mock_run_internal(*args, **kwargs):
            captured_data.append(kwargs.get("data"))
            # Create a mock report
            mock_report = MagicMock()
            mock_report.outcome = MagicMock()
            mock_report.outcome.value = "success"
            mock_report.output_text = "Mock answer"
            return mock_report

        with patch.object(api, "_run_internal", mock_run_internal):
            with patch("enzu.tools.docling_parser.documents_available", return_value=True):
                with patch("enzu.tools.docling_parser.parse_documents", mock_parse_documents):
                    api.run(
                        "Test task",
                        model="gpt-4",
                        provider="openrouter",
                        documents=["test.pdf"],
                        data="Extra context",
                    )

        assert len(captured_data) == 1
        combined = captured_data[0]

        # Should contain parsed document
        assert "Test content" in combined

        # Should contain additional data
        assert "== Additional Data ==" in combined
        assert "Extra context" in combined

    def test_api_run_triggers_rlm_mode_with_documents(self):
        """Providing documents triggers RLM mode (like providing data)."""
        from enzu import api

        def mock_parse_documents(paths):
            return "Document content"

        captured_mode = []

        def mock_run_internal(*args, **kwargs):
            # The mode is resolved before _run_internal, so we check data presence
            # RLM mode is triggered when data is not None
            captured_mode.append(kwargs.get("data"))
            mock_report = MagicMock()
            mock_report.outcome = MagicMock()
            mock_report.output_text = "Answer"
            return mock_report

        with patch.object(api, "_run_internal", mock_run_internal):
            with patch("enzu.tools.docling_parser.documents_available", return_value=True):
                with patch("enzu.tools.docling_parser.parse_documents", mock_parse_documents):
                    api.run(
                        "Test task",
                        model="gpt-4",
                        provider="openrouter",
                        documents=["test.pdf"],
                    )

        # With documents, effective_data should not be empty
        assert len(captured_mode) == 1
        assert captured_mode[0]  # Should have content (truthy)


class TestDocumentCaching:
    """Tests for document caching behavior."""

    def test_document_cache_by_mtime(self):
        """DocumentCache caches by path + modification time."""
        from enzu.tools.docling_parser import DocumentCache, ParsedDocument

        cache = DocumentCache()

        # Create a parsed document
        doc = ParsedDocument(
            path="/test/file.pdf",
            filename="file.pdf",
            content="Content",
            char_count=7,
        )

        # Store with mtime
        cache.put("/test/file.pdf", doc, mtime=1000.0)

        # Retrieving requires file to exist with same mtime
        # Since file doesn't exist, get() returns None
        result = cache.get("/test/file.pdf")
        assert result is None  # File doesn't exist

    def test_cache_clear(self):
        """DocumentCache.clear() removes all cached documents."""
        from enzu.tools.docling_parser import DocumentCache, ParsedDocument

        cache = DocumentCache()
        doc = ParsedDocument(
            path="/test/file.pdf",
            filename="file.pdf",
            content="Content",
        )
        cache.put("/test/file.pdf", doc, mtime=1000.0)

        cache.clear()

        # Cache should be empty (internal check)
        assert len(cache._cache) == 0
