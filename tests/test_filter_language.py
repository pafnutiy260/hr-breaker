"""Tests for language parameter in filter chain."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource, FilterResult, ValidationResult
from hr_breaker.models.language import get_language


class TestBaseFilterLanguageParam:
    """All filters accept language parameter."""

    @pytest.fixture
    def job(self):
        return JobPosting(
            title="Backend Engineer", company="Acme",
            requirements=["Python"], keywords=["python"],
        )

    @pytest.fixture
    def source(self):
        return ResumeSource(content="John Doe\nPython dev")

    @pytest.fixture
    def optimized(self, source):
        return OptimizedResume(
            html="<div>Test</div>", source_checksum=source.checksum,
            pdf_text="Test", pdf_bytes=b"pdf",
        )

    def test_content_length_checker_accepts_language(self, optimized, job, source):
        """ContentLengthChecker.evaluate accepts language param."""
        from hr_breaker.filters import ContentLengthChecker
        f = ContentLengthChecker()
        import asyncio
        asyncio.run(f.evaluate(optimized, job, source, language=get_language("ru")))

    def test_data_validator_accepts_language(self, optimized, job, source):
        """DataValidator.evaluate accepts language param."""
        from hr_breaker.filters import DataValidator
        f = DataValidator()
        import asyncio
        asyncio.run(f.evaluate(optimized, job, source, language=get_language("ru")))


class TestRunFiltersLanguage:
    """run_filters passes language to each filter."""

    @pytest.mark.asyncio
    async def test_language_passed_to_filters(self):
        """run_filters should pass language to filter.evaluate."""
        russian = get_language("ru")
        source = ResumeSource(content="John Doe\nPython dev")
        optimized = OptimizedResume(
            html="<div>Test</div>", source_checksum=source.checksum,
            pdf_text="Test", pdf_bytes=b"pdf",
        )
        job = JobPosting(
            title="Dev", company="Co",
            requirements=["Python"], keywords=["python"],
        )

        captured_language = []

        class MockFilter:
            name = "MockFilter"
            priority = 1
            threshold = 0.5
            def __init__(self, no_shame=False):
                pass
            async def evaluate(self, optimized, job, source, language=None):
                captured_language.append(language)
                return FilterResult(
                    filter_name="MockFilter", passed=True, score=1.0,
                )

        with patch("hr_breaker.orchestration.FilterRegistry") as mock_registry:
            mock_registry.all.return_value = [MockFilter]

            from hr_breaker.orchestration import run_filters
            await run_filters(optimized, job, source, language=russian)

            assert captured_language == [russian]
