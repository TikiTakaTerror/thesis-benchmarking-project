"""LTN family adapter stub."""

from __future__ import annotations

from ..base import PhaseStubModelAdapter


class LTNModelAdapter(PhaseStubModelAdapter):
    """Shared interface stub for the LTNtorch model family."""

    family_name = "ltn"
    implementation_phase = 6

