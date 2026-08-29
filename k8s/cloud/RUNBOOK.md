# One cloud GPU node: create it, use it, destroy it in the same sitting

**Status: not executed.** Everything in this directory is ready to run and nothing here
has been run. No cloud resource has been created, and no bill has been incurred. Two
things are missing and only one of them is technical:

1. **No cloud credentials are configured on this machine.** `gcloud` is installed but has
   no authenticated account and no project set; the OCI and AWS CLIs are not installed.
2. **No spending cap has been agreed.** Creating a paid GPU instance is not something to
   do on someone's behalf on inference. The first section below is the conversation that
   has to happen first, not paperwork to skim.

Read this top to bottom before the first command. The order is the point: the billing
alert goes up **before** the instance, and the teardown is verified **before** the session
ends.

---

## 0. Agree the cap first

Fill this in, out loud, before anything is created:

| | |
|---|---|
| Hard ceiling for this exercise | `$______` (a low two figure number is enough) |
| Instance type and region | |
| On-demand or spot/preemptible | spot/preemptible unless there is a reason |
| Expected wall clock | 60-90 minutes including setup and teardown |
| Who tears it down, and when | **the same person, the same session** |

The standing rule this sits under: no four figure spending on career work. This exercise
is a low two figure number or it does not happen. A GPU instance left running overnight
turns a $12 exercise into a $300 one, and it does it silently.

**The instance is destroyed in the session that created it. There is no version of this
where it is left up "just until tomorrow".**

## 1. Billing alert, before the instance exists

An alert configured after the instance is an alert that did not cover the window you
actually needed it for.

**OCI** (the natural pick: three OCI certifications become concrete rather than
decorative):

```bash
# Console: Billing & Cost Management -> Budgets -> Create Budget
#   Scope: the compartment this exercise will use, and nothing else
#   Amount: the hard ceiling from section 0
#   Alert rule: 50% actual, then 90% actual, to an address that is read today
oci budgets budget list --compartment-id "$OCI_COMPARTMENT" --output table
```

**GCP** (fine if OCI GPU quota is a fight, which it often is on a new tenancy):

```bash
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="mlsyslab-gpu-node" \
  --budget-amount="${CAP}USD" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9
```

Confirm the alert exists and lists your address before continuing. A budget with no
notification channel is a dashboard, not an alarm.

## 2. Check quota before creating anything

A GPU quota rejection after the node is half configured wastes the window. Ask first:

```bash
# OCI: GPU shapes need a service limit increase on a new tenancy. This can take days.
oci limits value list --compartment-id "$OCI_TENANCY" \
  --service-name compute --availability-domain "$AD" \
  --query "data[?contains(name, 'gpu')]" --output table

# GCP: preemptible GPU quota is per region and separate from the on-demand one.
gcloud compute regions describe "$REGION" \
  --format="table(quotas.metric, quotas.limit, quotas.usage)" | grep -i gpu
```

## 3. Create the node

Spot or preemptible. The workload is a benchmark that resumes: `already_done()` makes a
preempted sweep pick up exactly where it stopped, so the cheap instance class costs
nothing but a restart.

**OCI**, VM.GPU.A10.1 or similar:

```bash
oci compute instance launch \
  --compartment-id "$OCI_COMPARTMENT" \
  --availability-domain "$AD" \
  --shape VM.GPU.A10.1 \
  --display-name mlsyslab-gpu \
  --image-id "$UBUNTU_2404_GPU_IMAGE" \
  --subnet-id "$SUBNET" \
  --ssh-authorized-keys-file ~/.ssh/id_ed25519.pub \
  --preemptible-instance-config '{"preemptionAction":{"type":"TERMINATE","preserveBootVolume":false}}'
```

**GCP**, one L4:

```bash
gcloud compute instances create mlsyslab-gpu \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124-ubuntu-2204 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB
```

`--instance-termination-action=DELETE` matters. Without it a preempted spot instance
stops rather than being deleted, and a stopped instance still bills for its disk.

**Write the instance id down now**, in the same place as the teardown command. The
commonest way an instance survives the night is that nobody could remember what it was
called.

## 4. Join it to the cluster as a node

The control plane is k3s on the laptop. The GPU box joins as an agent:

```bash
# On the control plane:
sudo cat /var/lib/rancher/k3s/server/node-token

# On the GPU node (K3S_URL must be an address the node can actually reach; on a home
# lab that usually means a tailnet address rather than a LAN one):
curl -sfL https://get.k3s.io | K3S_URL="https://$CONTROL_PLANE:6443" \
  K3S_TOKEN="$NODE_TOKEN" sh -s - agent --node-label mlsyslab.gpu=true

# NVIDIA container toolkit, so containerd can hand the GPU to a pod:
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd \
  --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
sudo systemctl restart k3s-agent

# And the device plugin, which is what makes nvidia.com/gpu a schedulable resource:
kubectl apply -f k8s/cloud/nvidia-device-plugin.yaml
kubectl get nodes -o wide
kubectl describe node mlsyslab-gpu | grep -A3 'Allocatable'   # expect nvidia.com/gpu: 1
```

Sanity check before spending measurement time on it:

```bash
kubectl apply -f k8s/cloud/pod-gpu-agent.yaml
kubectl -n mlsyslab wait --for=condition=Ready pod/mlsyslab-gpu --timeout=600s
kubectl -n mlsyslab exec mlsyslab-gpu -c agent -- nvidia-smi
```

If `nvidia-smi` does not run inside the pod, stop and fix that before going further. A
sweep against a pod that silently fell back to CPU produces numbers that look plausible
and are worthless.

## 5. Run the sweep

```bash
mlsys probe --config configs/cloud-gpu.yaml
mlsys run configs/cloud-gpu.yaml --concurrent --prometheus-port 9109
```

`configs/cloud-gpu.yaml` puts the cloud node in its **own** resource group. It is a
different machine, so it runs concurrently with the laptop rather than taking turns with
it; that is the whole reason the scheduler distinguishes the two cases.

Expect the clock check to say something. A fresh cloud instance and a laptop are rarely
in agreement to the second, and the sweep records the offset either way.

## 6. Destroy it, and prove it

This is not the last item on a list. It is the reason the list has an order.

```bash
bash k8s/cloud/teardown.sh
```

The script deletes the node from the cluster, terminates the instance, then **queries the
provider again and fails loudly if anything is still there**. Do not take a delete
command's exit code as evidence; take the second query's empty result.

Then, by hand, because the script cannot see them:

- [ ] Boot/attached disks gone (they bill after the instance is terminated)
- [ ] Reserved static IP released, if one was created
- [ ] Console shows zero running instances in the region
- [ ] Budget page shows the spend for the session, and it is under the cap

## What to write down afterwards

- The instance shape, region, spot or on demand, and the actual cost.
- The GPU's `compute_capability`, `enforced_power_limit_W` and `ecc_mode` from the run
  record. A rented GPU can arrive power capped or with ECC enabled, and neither is
  visible from the card's name. The schema already carries all three, which is why.
- Whether the numbers agree with the laptop's RTX 3050 where the workloads overlap.
- The clock offset, which on a fresh cloud instance is the one that actually matters.
