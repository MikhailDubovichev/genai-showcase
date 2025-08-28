import json
from pathlib import Path


def test_templates_have_required_keys():
    base = Path(__file__).resolve().parents[1] / "config" / "templates"
    for name in ["config.nebius.json", "config.openai.json"]:
        data = json.loads((base / name).read_text(encoding="utf-8"))
        assert "llm" in data and isinstance(data["llm"], dict)
        assert "provider" in data["llm"]
        assert "base_url" in data["llm"]
        # In cloud, model names/settings may be under llm.model and embeddings.name (MVP kept minimal)
        # Just assert core keys exist; detailed shape is validated at runtime by factories.

