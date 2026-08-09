import json

import pytest

from mlsyslab.config import ConfigError, load, specs_to_table


def write_config(tmp_path, payload):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


BASE = {
    "experiment": "exp",
    "devices": {"laptop": {"kind": "local"}, "pi": {"host": "1.2.3.4", "user": "u"}},
    "models": {
        "small": {"quantization": "Q4_K_M",
                  "paths": {"laptop": "C:/m/small.gguf", "pi": "~/m/small.gguf"}},
    },
    "defaults": {"backend": "llamacpp", "repetitions": 3},
}


def test_matrix_expands_cartesian_product(tmp_path):
    payload = dict(BASE)
    payload["matrix"] = [{
        "devices": ["laptop", "pi"], "models": ["small"],
        "modes": ["throughput"], "threads": [1, 4], "prompt_tokens": [128],
        "output_tokens": [64],
    }]
    config = load(write_config(tmp_path, payload))
    assert len(config.specs) == 4  # 2 devices x 2 thread counts
    assert {s.device_id for s in config.specs} == {"laptop", "pi"}


def test_model_paths_resolve_per_device(tmp_path):
    payload = dict(BASE)
    payload["matrix"] = [{"devices": ["laptop", "pi"], "models": ["small"],
                          "modes": ["throughput"], "output_tokens": [64]}]
    config = load(write_config(tmp_path, payload))
    paths = {s.device_id: s.model_path for s in config.specs}
    assert paths["laptop"] == "C:/m/small.gguf"
    assert paths["pi"] == "~/m/small.gguf"
    assert all(s.quantization == "Q4_K_M" for s in config.specs)


def test_unknown_model_is_an_error(tmp_path):
    payload = dict(BASE)
    payload["matrix"] = [{"devices": ["laptop"], "models": ["typo"]}]
    with pytest.raises(ConfigError, match="unknown model 'typo'"):
        load(write_config(tmp_path, payload))


def test_unknown_device_is_an_error(tmp_path):
    payload = dict(BASE)
    payload["matrix"] = [{"devices": ["nope"], "models": ["small"]}]
    with pytest.raises(ConfigError, match="unknown device 'nope'"):
        load(write_config(tmp_path, payload))


def test_duplicate_points_are_an_error(tmp_path):
    payload = dict(BASE)
    payload["matrix"] = [
        {"devices": ["laptop"], "models": ["small"], "threads": [4]},
        {"devices": ["laptop"], "models": ["small"], "threads": [4]},
    ]
    with pytest.raises(ConfigError, match="identical"):
        load(write_config(tmp_path, payload))


def test_missing_matrix_is_an_error(tmp_path):
    payload = dict(BASE)
    with pytest.raises(ConfigError, match="matrix"):
        load(write_config(tmp_path, payload))


def test_model_without_path_for_device_is_an_error(tmp_path):
    payload = dict(BASE)
    payload["models"] = {"small": {"paths": {"laptop": "C:/m/small.gguf"}}}
    payload["matrix"] = [{"devices": ["pi"], "models": ["small"]}]
    with pytest.raises(ConfigError, match="no path for device 'pi'"):
        load(write_config(tmp_path, payload))


def test_dry_run_table_hides_tokens_for_vision(tmp_path):
    payload = dict(BASE)
    payload["defaults"] = {"backend": "onnxruntime"}
    payload["models"] = {"cnn": {"paths": {"laptop": "C:/m/cnn.onnx"}}}
    payload["matrix"] = [{"devices": ["laptop"], "models": ["cnn"], "batch_sizes": [1]}]
    config = load(write_config(tmp_path, payload))
    table = specs_to_table(config.specs)
    assert "prompt" not in table
    assert "batch" in table
