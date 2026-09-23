"""Wikipedia evidence survives weak worker and supervisor final answers."""

import json

from app.orchestration.state import AgentState, ToolCallRecord
from app.tools.base import ToolResult


def test_successful_wikipedia_lookup_overrides_false_absence() -> None:
    state = AgentState(user_request="Muhammed Salah kimdir?")
    state.tool_results.append(ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Muhammed Salah"},
        result=ToolResult.ok(json.dumps({
            "title": "Muhammed Salah", "extract": "Mısırlı futbolcudur.",
            "url": "https://tr.wikipedia.org/wiki/Muhammed_Salah",
        })),
    ))
    state.finish("Türkçe Wikipedia'da bir makale bulunamadığı için bilgi veremem.")
    assert "maddesi bulundu" in state.final_answer
    assert "Mısırlı futbolcudur" in state.final_answer
    assert "https://tr.wikipedia.org/wiki/Muhammed_Salah" in state.final_answer


def test_successful_wikipedia_lookup_adds_missing_citation() -> None:
    state = AgentState(user_request="Muhammed Salah kimdir?")
    state.tool_results.append(ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Muhammed Salah"},
        result=ToolResult.ok(json.dumps({
            "title": "Muhammed Salah", "extract": "Mısırlı futbolcudur.",
            "url": "https://tr.wikipedia.org/wiki/Muhammed_Salah",
        })),
    ))
    state.finish("Muhammed Salah Mısırlı futbolcudur.")
    assert state.final_answer is not None
    assert state.final_answer.endswith("https://tr.wikipedia.org/wiki/Muhammed_Salah")


def test_english_source_keeps_original_intro_when_model_changes_nationality() -> None:
    state = AgentState(user_request="Mo Salah (footballer, born 2004) kimdir?")
    state.tool_results.append(ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Mo Salah (footballer, born 2004)"},
        result=ToolResult.ok(json.dumps({
            "title": "Mo Salah (footballer, born 2004)",
            "extract": "Mohamed Salah El Boukammiri (born 27 May 2004) is a "
                       "Belgian professional footballer.",
            "url": "https://en.wikipedia.org/wiki/Mo_Salah_%28footballer%2C_born_2004%29",
        })),
    ))
    state.finish("Birleşik Krallık futbolcusudur.")
    assert state.final_answer is not None
    assert "Belgian professional footballer" in state.final_answer
    assert "Birleşik Krallık" not in state.final_answer
