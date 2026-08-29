# Manifests

`K8sDevice` generates and applies an equivalent pod from the device config, so for the
normal path you do not need anything in this directory. The limits that change a
measurement belong in the same file as the experiment that produced them, not in a
separate YAML that will eventually disagree with it.

These are here for two reasons:

* to read, when you want to know exactly what the device creates; and
* for `create_pod: false`, when you would rather manage pod lifecycle yourself.

| file | what it is |
|---|---|
| `namespace.yaml` | the `mlsyslab` namespace |
| `pod-agent-unlimited.yaml` | an agent pod with no CPU quota |
| `pod-agent-cpu4.yaml` | the same pod with a 4 core CFS quota, the constrained arm |
| `cloud/` | the single cloud GPU node, and the teardown that must follow it |

Every generated pod carries an `mlsyslab/spec-hash` annotation. If the device finds a pod
whose hash does not match the config it is holding, it deletes and recreates it, because
most of a pod spec is immutable and running a sweep against yesterday's CPU quota while
labelling it with today's is the exact failure this repository exists to avoid. A pod that
carries no hash at all is treated as foreign and is never adopted.

## Applying by hand

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pod-agent-cpu4.yaml
kubectl wait --for=condition=Ready pod/mlsyslab-cpu4 -n mlsyslab --timeout=300s
```

Then set `create_pod: false` on the device so the harness uses what you started rather
than replacing it. Note that with `create_pod: false` a mismatch between the pod and the
config becomes an error rather than a recreation: the harness will not measure through a
pod whose limits it cannot vouch for.
