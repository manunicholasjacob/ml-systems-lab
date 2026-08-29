"""Regenerate the checked-in pod manifests from the device that creates them.

The manifests in ``k8s/`` exist to be read. They are generated rather than written by
hand so that the pod a reader sees is provably the pod the harness applies: a
hand-maintained copy drifts, and a drifted manifest is worse than none, because it
describes limits that never applied to any number in the results directory.

    python tools/write_manifests.py
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from mlsyslab.devices import from_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = {
    "kind": "k8s",
    "namespace": "mlsyslab",
    "image": "docker.io/library/mlsyslab-agent:0.2.0",
    "memory_request": "2Gi",
    "memory_limit": "4Gi",
    "host_mounts": [{"host_path": "/home/manu/models", "mount_path": "/models",
                     "read_only": True}],
}

VARIANTS = [
    ("pod-agent-unlimited.yaml", "k8s-unlimited", "mlsyslab-unlimited", {},
     "No CPU quota. Isolates containerisation and orchestration on their own:\n"
     "# namespaces, overlayfs, CNI and a pod sandbox, with the scheduler out of the way."),
    ("pod-agent-cpu4.yaml", "k8s-cpu4", "mlsyslab-cpu4",
     {"cpu_limit": "4", "cpu_request": "4"},
     "A 4 core CFS quota, equal to the benchmark's thread count. This is the arm\n"
     "# that answers whether a quota you would actually set in production costs you\n"
     "# throughput even when average utilisation sits inside it."),
]

HEADER = """# GENERATED from the device config by K8sDevice.pod_manifest(), and checked in so
# the pod a reader sees is the pod the harness creates. Regenerate with
# tools/write_manifests.py rather than editing by hand.
#
# {why}
#
# The mlsyslab/spec-hash annotation is how the device notices that a running pod no
# longer matches the config it is holding. Do not remove it: a pod without one is
# treated as foreign and will not be adopted.
"""


def dump(manifest: dict) -> str:
    try:
        import yaml
    except ImportError:
        # JSON is valid YAML, and the core of this package has no hard YAML dependency.
        return json.dumps(manifest, indent=2) + "\n"
    return yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)


def main() -> int:
    written = []
    for filename, device_id, pod, extra, why in VARIANTS:
        config = dict(BASE, pod=pod, **extra)
        manifest = from_config(device_id, config).pod_manifest()
        path = os.path.join(REPO, "k8s", filename)
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(HEADER.format(why=why) + dump(manifest))
        written.append(filename)
    print("wrote " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
