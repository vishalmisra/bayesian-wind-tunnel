"""
J-lens experiments: running Anthropic's global-workspace Jacobian lens
inside the Bayesian wind tunnel.

Tests whether the emergent "J-space" (Anthropic, "A global workspace in
language models", 2026; code: anthropics/jacobian-lens @ 581d3986) and
Paper I's hypothesis frame are the same object.

Preregistered predictions P1-P5: see the experiment spec and DEVIATIONS.md.
"""

ANTHROPIC_JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
