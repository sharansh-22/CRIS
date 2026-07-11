"""Explicit error types for the CRIS Data Layer."""

from __future__ import annotations


class DataLayerError(Exception):
    """Base class for data-layer failures."""


class DataSourceError(DataLayerError):
    """Raised when a provider cannot be reached or returns unusable data."""


class DatasetContractError(DataLayerError):
    """Raised when a dataset violates schema or semantic contract rules."""


class DataPublishError(DataLayerError):
    """Raised when cleaned data cannot be safely published."""
