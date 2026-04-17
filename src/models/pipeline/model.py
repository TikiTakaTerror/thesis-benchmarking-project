"""Pipeline family adapter stub."""

from __future__ import annotations

from ..base import PhaseStubModelAdapter


class PipelineModelAdapter(PhaseStubModelAdapter):
    """Shared interface stub for the custom symbolic pipeline family."""

    family_name = "pipeline"
    implementation_phase = 4

