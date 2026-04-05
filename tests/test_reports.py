"""Report generation endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_generate_report(client: AsyncClient):
    resp = await client.post(
        "/api/v1/reports/generate",
        json={
            "report_type": "weekly",
            "date_range_start": "2026-03-28",
            "date_range_end": "2026-04-04",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["report_type"] == "weekly"
    assert data["json_summary"] is not None
    summary = data["json_summary"]
    assert "meals_logged" in summary
    assert "symptom_summary" in summary
    assert "top_triggers" in summary
