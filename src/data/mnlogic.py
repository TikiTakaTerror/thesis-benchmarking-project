"""MNLogic dataset adapter."""

from __future__ import annotations

from .prepared import PreparedManifestDatasetAdapter


class MNLogicDatasetAdapter(PreparedManifestDatasetAdapter):
    """Dataset adapter for MNLogic prepared in the local manifest contract."""

    @property
    def name(self) -> str:
        return "mnlogic"

