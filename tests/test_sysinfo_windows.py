"""Windows machine description without wmic, which Windows 11 24H2 no longer ships."""

from mlsyslab import sysinfo


def test_registry_brand_string_is_used_first(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sysinfo, "_windows_cpu_name",
                        lambda: "12th Gen Intel(R) Core(TM) i7-12700H")
    assert sysinfo.cpu_model() == "12th Gen Intel(R) Core(TM) i7-12700H"


def test_identifier_is_the_last_resort(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sysinfo, "_windows_cpu_name", lambda: None)
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: None)
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "Intel64 Family 6 Model 154")
    assert sysinfo.cpu_model() == "Intel64 Family 6 Model 154"


def test_core_count_and_gpu_fall_back_to_cim(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] in ("wmic", "nvidia-smi"):
            return None
        if "NumberOfCores" in cmd[-1]:
            return "14\r\n"
        if "Win32_VideoController" in cmd[-1]:
            return "NVIDIA GeForce RTX 3050 Laptop GPU\r\nIntel(R) Iris(R) Xe Graphics\r\n"
        return None

    monkeypatch.setattr(sysinfo, "_run", fake_run)
    assert sysinfo.physical_cores() == 14
    assert sysinfo.gpu() == "NVIDIA GeForce RTX 3050 Laptop GPU"
