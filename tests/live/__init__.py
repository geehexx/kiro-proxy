"""Live smoke tests — exercise a running kiro-gateway end-to-end.

Only run when the user explicitly asks for it:
    KIRO_GATEWAY_URL=http://127.0.0.1:8765 \
    KIRO_GATEWAY_API_KEY=<key> \
    .venv/bin/python -m pytest tests/live -v -m live

Excluded from the default pytest run by the `live` marker in
pytest.ini (see ``markers``).
"""
