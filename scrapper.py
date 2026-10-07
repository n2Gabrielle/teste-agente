"""Compatibilidade para o módulo de scraping anterior."""

from services.scraper import ScrapingError, get_text_from_url


__all__ = ["ScrapingError", "get_text_from_url"]