"""Kand-Logic dataset adapter."""

from __future__ import annotations

from .prepared import PreparedManifestDatasetAdapter


class KandLogicDatasetAdapter(PreparedManifestDatasetAdapter):
    """Dataset adapter for prepared Kand-Logic manifests."""

    @property
    def name(self) -> str:
        return "kand_logic"
