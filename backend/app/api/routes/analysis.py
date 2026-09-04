"""Placeholder module.

The document screening / analysis endpoint for Phase 1 + Phase 2 lives in
``app.api.routes.documents`` (``POST /api/screen``). That route delegates to
``app.pipeline.pipeline.ScreeningPipeline``, which runs Phase 1 (document/OCR)
and then the Phase 2 Rules Engine, returning both in a single
``ScreeningResponse`` (the rule findings are on ``response.rules``).

No separate analysis route is added: adding one would duplicate the pipeline
invocation. This file is intentionally left without a router to avoid a second,
divergent processing path.
"""
