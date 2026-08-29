#!/usr/bin/env bash
# Destroy the cloud GPU node, then prove it is gone.
#
# The proof is the point. A delete command that exits zero has told you the request was
# accepted, not that the resource stopped billing. Every check below re-queries the
# provider after the delete and fails loudly if anything answers.
#
#   PROVIDER=gcp ZONE=us-central1-a INSTANCE=mlsyslab-gpu bash k8s/cloud/teardown.sh
#   PROVIDER=oci INSTANCE_OCID=ocid1.instance.oc1... bash k8s/cloud/teardown.sh
#
# Run this in the same session that created the node. Not tomorrow.

set -uo pipefail

PROVIDER="${PROVIDER:-}"
INSTANCE="${INSTANCE:-mlsyslab-gpu}"
NODE="${NODE:-$INSTANCE}"
KUBECTL="${KUBECTL:-kubectl}"
failures=0

say()  { printf '\n== %s\n' "$*"; }
bad()  { printf '   FAIL: %s\n' "$*"; failures=$((failures + 1)); }
good() { printf '   ok: %s\n' "$*"; }

# ---------------------------------------------------------------- 1. leave the cluster
say "Removing $NODE from the cluster"
$KUBECTL delete pod mlsyslab-gpu -n mlsyslab --ignore-not-found --wait=true >/dev/null 2>&1
$KUBECTL drain "$NODE" --ignore-daemonsets --delete-emptydir-data --force \
  --timeout=120s >/dev/null 2>&1
$KUBECTL delete node "$NODE" --ignore-not-found >/dev/null 2>&1
if $KUBECTL get node "$NODE" >/dev/null 2>&1; then
  bad "node $NODE is still registered with the cluster"
else
  good "node $NODE is no longer in the cluster"
fi

# ------------------------------------------------------------- 2. destroy the instance
case "$PROVIDER" in
  gcp)
    : "${ZONE:?set ZONE}"
    say "Deleting GCP instance $INSTANCE in $ZONE"
    gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
    # The re-query, which is the only thing that counts.
    if gcloud compute instances describe "$INSTANCE" --zone="$ZONE" >/dev/null 2>&1; then
      bad "instance $INSTANCE still exists and is still billing"
    else
      good "instance $INSTANCE is gone"
    fi
    say "Disks left behind (these bill after the instance is deleted)"
    orphans=$(gcloud compute disks list --filter="zone:($ZONE) AND -users:*" \
                --format="value(name)" 2>/dev/null)
    if [ -n "$orphans" ]; then
      bad "unattached disks remain: $orphans"
      printf '   delete with: gcloud compute disks delete %s --zone=%s\n' "$orphans" "$ZONE"
    else
      good "no unattached disks in $ZONE"
    fi
    say "Reserved addresses (a held static IP bills while unused)"
    held=$(gcloud compute addresses list --filter="status=RESERVED" \
             --format="value(name)" 2>/dev/null)
    [ -n "$held" ] && bad "reserved addresses remain: $held" \
                   || good "no reserved addresses"
    say "Anything still running in this project"
    gcloud compute instances list --format="table(name,zone,status)" 2>/dev/null
    ;;

  oci)
    : "${INSTANCE_OCID:?set INSTANCE_OCID}"
    say "Terminating OCI instance $INSTANCE_OCID"
    oci compute instance terminate --instance-id "$INSTANCE_OCID" \
      --preserve-boot-volume false --force --wait-for-state TERMINATED
    state=$(oci compute instance get --instance-id "$INSTANCE_OCID" \
              --query 'data."lifecycle-state"' --raw-output 2>/dev/null)
    if [ "$state" = "TERMINATED" ] || [ -z "$state" ]; then
      good "instance state is ${state:-gone}"
    else
      bad "instance state is $state, not TERMINATED"
    fi
    say "Boot volumes left behind"
    if [ -n "${OCI_COMPARTMENT:-}" ] && [ -n "${AD:-}" ]; then
      oci bv boot-volume list --compartment-id "$OCI_COMPARTMENT" \
        --availability-domain "$AD" \
        --query "data[?\"lifecycle-state\"=='AVAILABLE'].{name:\"display-name\",id:id}" \
        --output table 2>/dev/null
      printf '   any AVAILABLE boot volume above is unattached and still billing\n'
    else
      bad "set OCI_COMPARTMENT and AD to check for orphaned boot volumes"
    fi
    say "Anything still running in this compartment"
    [ -n "${OCI_COMPARTMENT:-}" ] && oci compute instance list \
      --compartment-id "$OCI_COMPARTMENT" \
      --query "data[?\"lifecycle-state\"!='TERMINATED'].{name:\"display-name\",state:\"lifecycle-state\"}" \
      --output table 2>/dev/null
    ;;

  *)
    bad "set PROVIDER=gcp or PROVIDER=oci. Refusing to guess which cloud is billing you."
    ;;
esac

say "Result"
if [ "$failures" -eq 0 ]; then
  printf '   Everything this script can see is gone.\n'
  printf '   Still check by hand, because this script cannot see them:\n'
  printf '     - snapshots and machine images\n'
  printf '     - the budget page, which should show the session spend under the cap\n'
  exit 0
fi
printf '   %d check(s) failed. Something is still billing. Do not close this session.\n' \
  "$failures"
exit 1
