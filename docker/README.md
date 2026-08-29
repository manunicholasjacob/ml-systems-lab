# Building the agent image, and standing up a cluster to run it on

Everything here is four commands and a `kubectl`. None of it is optional if you want the
containerisation numbers in `results/containerization/` to mean anything, because the
comparison depends on the host and the pod running the *same binary*.

## 1. Build the image

```bash
docker build -t mlsyslab-agent:0.2.0 -f docker/Dockerfile .
```

The build compiles llama.cpp from a pinned commit and copies the binaries into a
digest-pinned `python:3.11-slim`. The harness itself is **not** baked in: it is streamed
into the container at run time, exactly as the SSH device streams it to the Pi, so a code
change does not need a rebuild.

`GGML_NATIVE` defaults to `ON`, which makes the image specific to the CPU family it was
built on. That is the correct trade for a benchmark and the wrong one for distribution.
For a portable image:

```bash
docker build --build-arg GGML_NATIVE=OFF -t mlsyslab-agent:0.2.0-portable -f docker/Dockerfile .
```

## 2. Load it into the cluster

k3s uses containerd, not the Docker daemon, so a locally built image has to be handed
over. No registry is involved:

```bash
docker save mlsyslab-agent:0.2.0 | sudo k3s ctr images import -
sudo k3s ctr images ls -q | grep mlsyslab
```

## 3. Extract the same binaries for the host arm

This is the step that makes the experiment a controlled comparison rather than two
different builds being compared to each other:

```bash
CID=$(docker create mlsyslab-agent:0.2.0)
mkdir -p ~/llama-bin
docker cp "$CID:/usr/local/bin/llama-bench"  ~/llama-bin/
docker cp "$CID:/usr/local/bin/llama-server" ~/llama-bin/
docker cp "$CID:/usr/local/bin/llama-cli"    ~/llama-bin/
docker cp "$CID:/usr/local/lib/mlsyslab/."   ~/llama-bin/
docker cp "$CID:/etc/mlsyslab-llamacpp-commit" ~/llama-bin/COMMIT
docker rm "$CID"
chmod +x ~/llama-bin/llama-*
sha256sum ~/llama-bin/llama-bench      # record this; it should match the pod's copy
```

The binaries are linked against shared objects that are on `LD_LIBRARY_PATH` inside the
image and are not on any search path outside it. Rather than patch the binary, which
would change the bytes and defeat the whole point, the device config names the directory:

```yaml
llamacpp:
  bin_dir: /home/manu/llama-bin
  env:
    LD_LIBRARY_PATH: /home/manu/llama-bin
```

## 4. Put the models on the node

The pods mount the model directory from the node read-only, so every arm reads the same
file and no overlayfs copy sits in the read path. Copy them onto the node's own
filesystem, not a network or `/mnt` mount, or you will be measuring your filesystem:

```bash
mkdir -p ~/models
cp /path/to/*.gguf ~/models/
```

## Standing up k3s

```bash
curl -sfL https://get.k3s.io | sudo INSTALL_K3S_EXEC="server \
  --write-kubeconfig-mode 644 --disable traefik --disable servicelb --disable metrics-server" sh -
sudo k3s kubectl get nodes
```

Traefik, the service load balancer and metrics-server are all disabled because none of
them are used here, and every one of them is a process competing for the CPU that is
being measured.

`k8s/` holds the manifests. You do not need to apply them by hand: `K8sDevice` generates
an equivalent pod from the device config and applies it, so the resource limits that
change the measurement live in the same file as the experiment that produced it. The
checked-in manifests are for reading, and for `create_pod: false` if you would rather
manage pods yourself.

## Windows and WSL2, which is where this was developed

Five things bite, in the order they bit.

**The k3s API server is not reachable from the Windows side.** Port 22 forwards through
WSL2's localhost relay and 6443 does not, on this machine, in either direction, whether
addressed as `127.0.0.1` or as the WSL adapter's address. Rather than fight it, the device
config makes `kubectl` an argv prefix:

```yaml
kubectl: ["wsl", "-d", "Ubuntu", "--exec", "k3s", "kubectl"]
```

**`--exec` is load-bearing.** `wsl -d Ubuntu -- <argv>` joins the arguments and hands the
result to a login shell, which re-parses them. This does not fail. It answers a different
question: `kubectl exec ... -- sh -c 'printf %s "$HOME"'` returned the *WSL host's* home
directory instead of the container's, and a `-o jsonpath` query reported a serving pod as
not Ready. `K8sDevice.verify_transport()` sends a string full of quotes, spaces, dollars
and parentheses through the transport and refuses to measure anything if it comes back
changed. With `--exec` there is no shell and nothing is re-parsed.

**Nothing reads cluster state as jsonpath.** For the same reason. `kubectl get pod -o json`
plus a dictionary lookup in Python has no characters for anything to chew on.

**WSL2's localhost relay is not dependable, and it fails by disappearing.** The host arm
of the containerisation study reaches the Linux side over SSH, and `127.0.0.1:22`
forwarded reliably for both passes of the sweep and then started refusing connections
with the VM still up, sshd still listening on `0.0.0.0:22`, and nothing in any log. The
distro's own address still worked:

```bash
wsl -d Ubuntu --exec hostname -I      # e.g. 172.17.105.249
```

Put that in the device's `host:` when the relay stops answering. It changes when the VM
restarts, which is the reason it is not the documented default. Port 6443 never forwarded
through the relay at all on this machine, in either direction, which is why the Kubernetes
device goes through `wsl --exec` instead of a kubeconfig.

**Set `vmIdleTimeout` before running anything long.** The WSL2 utility VM shuts itself
down when no `wsl.exe` process is attached, which during a sweep is most of the time: the
harness opens one connection per run and closes it again. The VM going away takes k3s,
containerd and every pod with it, and it does not look like a crash from outside. It looks
like pods reporting `SandboxChanged, it will be killed and re-created` every few minutes,
and a sweep that stops producing records without an error. In `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=-1
```

then `wsl --shutdown` to apply it. Restarted containers are handled anyway, since the
package sync is keyed to the container's lifetime rather than the session's, but a sweep
still runs faster when the cluster stays up.

## Verifying the setup

```bash
mlsys probe --config configs/containerization.yaml
mlsys run configs/containerization.yaml --dry-run
```

`probe` reports each device's CPU, capabilities and, for pods, the effective cgroup limits
read from *inside* the container rather than copied from the config. If a pod reports a
quota you did not ask for, believe the pod.

## Undoing all of it

Standing this up leaves things on the machine. Everything below is reversible, and
nothing here is needed once you are done measuring.

```bash
# Inside WSL: the cluster, its images, and the staged models
sudo /usr/local/bin/k3s-uninstall.sh     # removes k3s, containerd and every pod
sudo docker rmi mlsyslab-agent:0.2.0
rm -rf ~/llama-bin ~/models              # the extracted binaries and the GGUF files
sudo systemctl disable --now ssh docker  # if they were installed for this
```

```powershell
# On Windows: stop the VM from being pinned awake
#   delete the [wsl2] vmIdleTimeout=-1 block from %USERPROFILE%\.wslconfig
wsl --shutdown
```

The SSH key pair generated for the host arm is `~/.ssh/wsl_lab_key`; its public half is in
`~/.ssh/authorized_keys` inside the distro. Remove both if the host arm is not going to be
run again.

What is safe to leave: k3s idles at a few hundred megabytes and starts with the distro,
which is convenient if you intend to run more sweeps and pure overhead if you do not.
