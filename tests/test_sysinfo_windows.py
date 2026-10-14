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


def _cim_lists(monkeypatch, listing):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] in ("wmic", "nvidia-smi"):
            return None
        if "Win32_VideoController" in cmd[-1]:
            return listing
        return None

    monkeypatch.setattr(sysinfo, "_run", fake_run)


def test_discrete_gpu_wins_when_the_integrated_one_is_listed_first(monkeypatch):
    _cim_lists(monkeypatch,
               "Intel(R) Iris(R) Xe Graphics\r\nNVIDIA GeForce RTX 3050 Laptop GPU\r\n")
    assert sysinfo.gpu() == "NVIDIA GeForce RTX 3050 Laptop GPU"


def test_wmic_listing_also_prefers_the_discrete_gpu(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")

    def fake_run(cmd, timeout=5.0):
        if cmd[0] == "wmic":
            return ("\r\n\r\nName=AMD Radeon(TM) Graphics\r\n\r\n"
                    "Name=AMD Radeon RX 7600M XT\r\n\r\n")
        return None

    monkeypatch.setattr(sysinfo, "_run", fake_run)
    assert sysinfo.gpu() == "AMD Radeon RX 7600M XT"


def test_integrated_arc_and_radeon_names_are_not_mistaken_for_discrete():
    pick = sysinfo._pick_display_adapter
    assert pick(["Intel(R) Arc(TM) Graphics", "NVIDIA RTX 2000 Ada Generation Laptop GPU"]) \
        == "NVIDIA RTX 2000 Ada Generation Laptop GPU"
    assert pick(["AMD Radeon(TM) 780M Graphics", "Intel(R) Arc(TM) A770 Graphics"]) \
        == "Intel(R) Arc(TM) A770 Graphics"
    assert pick(["AMD Radeon(TM) 780M Graphics", "AMD Radeon Pro W7600"]) \
        == "AMD Radeon Pro W7600"


def test_virtual_adapters_lose_to_a_real_integrated_gpu(monkeypatch):
    _cim_lists(monkeypatch,
               "Microsoft Basic Display Adapter\r\nParsec Virtual Display Adapter\r\n"
               "Intel(R) UHD Graphics 770\r\n")
    assert sysinfo.gpu() == "Intel(R) UHD Graphics 770"


def test_listing_order_is_kept_between_equals(monkeypatch):
    _cim_lists(monkeypatch, "Intel(R) UHD Graphics 630\r\nIntel(R) HD Graphics 530\r\n")
    assert sysinfo.gpu() == "Intel(R) UHD Graphics 630"
    assert sysinfo._pick_display_adapter(["Microsoft Basic Display Adapter"]) \
        == "Microsoft Basic Display Adapter"
    assert sysinfo._pick_display_adapter(["", "  "]) is None
