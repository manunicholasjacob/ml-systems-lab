"""The GPU half of the device abstraction, tested without a GPU.

The framework's rule is that a machine cannot be a first-class device if measuring it
needs hardware in the room, and that applies to the tests too. Everything here runs off
recorded `nvidia-smi` output and stubbed imports, the same way the llama.cpp backend is
tested off recorded `llama-bench` output.

The one thing these cannot cover is whether the CUDA kernels in `membw.measure_device`
report a sane ceiling on a real card. That needs a real card and is unrun.
"""

import dataclasses

import pytest

from mlsyslab import membw, schema, sysinfo
from mlsyslab.telemetry import power

# Recorded from an RTX 3050 Laptop GPU on 23 August 2026. A consumer part is the useful
# fixture precisely because it refuses some queries: power.limit and ecc.mode come back
# "[N/A]" here and a datacenter part answers both, so this is the case that has to not
# turn absent into zero.
SMI_ROW = ("NVIDIA GeForce RTX 3050 Laptop GPU, 596.08, 8.6, 4096 MiB, 4, 8, "
           "40.00 W, [N/A], [N/A], 2100 MHz, 5501 MHz, [N/A]")

SMI_TWO_CARDS = (SMI_ROW + "\n"
                 + SMI_ROW.replace("NVIDIA GeForce RTX 3050 Laptop GPU", "Tesla T4")
                          .replace("4096 MiB", "15360 MiB"))


# ------------------------------------------------------------------ nvidia-smi parsing

def test_units_are_stripped_and_numbers_come_back_as_numbers():
    assert sysinfo._smi_value("4096 MiB") == 4096
    assert sysinfo._smi_value("40.00 W") == 40.0
    assert sysinfo._smi_value("5501 MHz") == 5501


def test_a_query_the_card_refuses_is_absent_not_zero():
    # A zero power limit or a zero clock would be read as a measurement. It is not one.
    for refusal in ("[N/A]", "[Not Supported]", "", "  "):
        assert sysinfo._smi_value(refusal) is None


def test_version_shaped_fields_stay_strings():
    # 8.10 must not sort below 8.6, which is what happens if this becomes a float.
    assert sysinfo._smi_value("8.6", "compute_capability") == "8.6"
    assert sysinfo._smi_value("596.08", "driver") == "596.08"


def test_gpus_parses_a_recorded_row(monkeypatch):
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: SMI_ROW)
    found = sysinfo.gpus()
    assert len(found) == 1
    gpu = found[0]
    assert gpu["index"] == 0
    assert gpu["name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu["compute_capability"] == "8.6"
    assert gpu["memory_total_MiB"] == 4096
    assert gpu["pcie_gen"] == 4 and gpu["pcie_width"] == 8
    assert gpu["enforced_power_limit_W"] == 40.0
    assert gpu["power_limit_W"] is None
    assert gpu["ecc_mode"] is None


def test_gpus_indexes_multiple_cards(monkeypatch):
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: SMI_TWO_CARDS)
    found = sysinfo.gpus()
    assert [g["index"] for g in found] == [0, 1]
    assert found[1]["name"] == "Tesla T4"
    assert found[1]["memory_total_MiB"] == 15360


def test_no_driver_means_an_empty_list_not_an_exception(monkeypatch):
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: None)
    assert sysinfo.gpus() == []


def test_a_short_row_is_skipped_rather_than_misaligned(monkeypatch):
    # nvidia-smi changing its output should lose the row, not shift every field by one.
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: "only, three, cells")
    assert sysinfo.gpus() == []


def test_collect_carries_the_gpu_list(monkeypatch):
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: SMI_ROW)
    assert sysinfo.collect()["gpus"][0]["compute_capability"] == "8.6"


# ------------------------------------------------------------------ power sources

def test_energy_counter_delta_is_plain_subtraction():
    assert power.nvml_delta_j(1_000, 3_500) == 2.5


def test_a_counter_that_went_backwards_reports_nothing_rather_than_guessing():
    # The NVML counter is 64-bit millijoules since driver load and does not wrap in any
    # practical window, so a decrease means the driver reloaded. There is no energy to
    # report for a window that straddles that.
    assert power.nvml_delta_j(5_000, 1_000) == 0.0


def test_the_energy_source_is_named_not_inferred(monkeypatch):
    monkeypatch.setattr(power, "read_nvml_energy_mj", lambda index=0: 1234)
    assert power.gpu_energy_source() == "nvml_energy_counter"

    monkeypatch.setattr(power, "read_nvml_energy_mj", lambda index=0: None)
    monkeypatch.setattr(power, "read_gpu_power_w", lambda index=0: 42.0)
    assert power.gpu_energy_source() == "nvidia_smi_instantaneous"

    monkeypatch.setattr(power, "read_gpu_power_w", lambda index=0: None)
    assert power.gpu_energy_source() is None


def test_nvml_absent_is_not_an_error(monkeypatch):
    monkeypatch.setattr(power, "_nvml", lambda: None)
    assert power.read_nvml_energy_mj() is None
    assert power.nvml_available() is False


def test_describe_reports_every_source_it_knows_about():
    described = power.describe()
    for key in ("pmic", "rapl", "nvml", "nvidia_smi", "gpu_energy_source"):
        assert key in described


def test_gpu_power_parses_the_bare_number(monkeypatch):
    monkeypatch.setattr(power, "_run_smi", lambda args: "8.07")
    assert power.read_gpu_power_w() == 8.07
    monkeypatch.setattr(power, "_run_smi", lambda args: "[N/A]")
    assert power.read_gpu_power_w() is None


# ------------------------------------------------------------------ device bandwidth

def test_no_torch_says_so_and_says_what_to_do(monkeypatch):
    monkeypatch.setattr(membw, "_torch", lambda: None)
    assert membw.device_available() is False
    with pytest.raises(membw.AcceleratorUnavailable) as caught:
        membw.measure_device()
    message = str(caught.value)
    assert "torch is not installed" in message
    assert "dram_peak_GBs" in message


def test_torch_without_cuda_is_distinguished_from_torch_missing(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(membw, "_torch", lambda: FakeTorch())
    with pytest.raises(membw.AcceleratorUnavailable) as caught:
        membw.measure_device()
    assert "no CUDA device" in str(caught.value)


def test_the_working_set_leaves_the_card_usable():
    class FakeCuda:
        @staticmethod
        def mem_get_info():
            return (8 * 1024 ** 3, 16 * 1024 ** 3)  # 8 GB free of 16

    class FakeTorch:
        cuda = FakeCuda()

    chosen = membw._device_working_set_bytes(FakeTorch(), None)
    assert chosen == membw.DEVICE_WORKING_SET_MB * 1024 * 1024
    # Asking for more than is free is capped, not honoured.
    capped = membw._device_working_set_bytes(FakeTorch(), 64 * 1024)
    assert capped <= 8 * 1024 ** 3 * (1 - membw.DEVICE_MIN_FREE_FRACTION)


def test_a_nearly_full_card_refuses_rather_than_measuring_cache():
    class FakeCuda:
        @staticmethod
        def mem_get_info():
            return (16 * 1024 * 1024, 16 * 1024 ** 3)  # 16 MB free

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.raises(membw.AcceleratorUnavailable) as caught:
        membw._device_working_set_bytes(FakeTorch(), None)
    assert "not enough to measure" in str(caught.value)


def test_the_device_path_keeps_the_host_path_s_stability_threshold():
    # One threshold, one meaning. A GPU on a shared host needs it more, not less.
    assert membw.STABILITY_THRESHOLD == 0.75


# ------------------------------------------------------------------ the record

def test_device_info_holds_the_gpu_list(monkeypatch):
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: SMI_ROW)
    known = {f.name for f in dataclasses.fields(schema.DeviceInfo)}
    device = schema.DeviceInfo(**{k: v for k, v in sysinfo.collect().items()
                                  if k in known})
    assert device.gpus[0]["memory_total_MiB"] == 4096
    assert device.gpu == device.gpus[0]["name"] or device.gpu is not None


def test_a_record_with_gpu_fields_round_trips():
    device = schema.DeviceInfo(gpus=[{"index": 0, "compute_capability": "7.5",
                                      "memory_total_MiB": 15360}])
    telemetry = schema.Telemetry(gpu_energy_j=12.5, gpu_power_w_mean=60.0,
                                 gpu_energy_per_token_mj=3.1,
                                 energy_source="nvml_energy_counter")
    record = schema.RunRecord(device=device, telemetry=telemetry)
    back = schema.from_dict(record.to_dict())
    assert back.device.gpus[0]["compute_capability"] == "7.5"
    assert back.telemetry.gpu_energy_j == 12.5
    assert back.telemetry.energy_source == "nvml_energy_counter"


def test_gpu_energy_is_not_folded_into_host_energy():
    # Two instruments with different error. Summing them would produce a number that
    # belongs to neither and cannot be checked against either.
    fields = {f.name for f in dataclasses.fields(schema.Telemetry)}
    assert "energy_j" in fields and "gpu_energy_j" in fields


def test_an_old_record_without_gpu_fields_still_loads():
    old = {"run_id": "abc", "device": {"cpu": "something", "kind": "local"},
           "telemetry": {"power_w_mean": 5.0}}
    back = schema.from_dict(old)
    assert back.device.gpus == []
    assert back.telemetry.gpu_energy_j is None
    assert back.telemetry.power_w_mean == 5.0
