import json
from pathlib import Path


def test_edge_templates_have_required_keys():
    base = Path(__file__).resolve().parents[1] / "config" / "templates"
    for name in ["edge_nebius.json", "edge_openai.json"]:
        data = json.loads((base / name).read_text(encoding="utf-8"))
        assert "llm" in data and isinstance(data["llm"], dict)
        assert "provider" in data["llm"]
        assert "base_url" in data["llm"]
        models = data["llm"].get("models", {})
        for key in ["classification", "device_control", "energy_efficiency"]:
            assert key in models and "settings" in models[key]

