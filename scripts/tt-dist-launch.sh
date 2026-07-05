#!/bin/bash
# Cross-node two-tower diffusion launcher. Run on BOTH nodes:
#   node .4 (den):  ROLE=den ~/tt-dist/tt-dist-launch.sh [extra args]
#   node .3 (ctx):  ROLE=ctx ~/tt-dist/tt-dist-launch.sh [extra args]
# Start den first (hosts the rendezvous), then ctx.
set -euo pipefail

ROLE=${ROLE:?set ROLE=ctx|den}
IMG=ghcr.io/aeon-7/aeon-vllm-ultimate:latest
MASTER=10.100.10.4
IFACE=enp1s0f1np1
NAME=tt-dist-${ROLE}

if [ "$ROLE" = "den" ]; then
  MODEL_DIR=/home/keyspark/aeon27b/models/tt-denoiser
else
  MODEL_DIR=/home/keyspark/aeon27b/models/tt-context
fi

# --- clear GPU (standing rule: always before any launch) ---
bash /home/keyspark/gpu-clear.sh || true
docker rm -f ${NAME} 2>/dev/null || true

# --- OOM fastkill watchdog ---
cat > /home/keyspark/fastkill-ttdist.sh <<EOF
#!/bin/bash
while :; do
  a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  if [ "\$a" -lt 3000000 ]; then
    docker rm -f ${NAME} 2>/dev/null
    echo "\$(date '+%F %T') FASTKILL fired at \${a}KB avail" >> /home/keyspark/oom-fastkill.log
    sleep 10
  fi
  sleep 2
done
EOF
chmod +x /home/keyspark/fastkill-ttdist.sh
pkill -f '[f]astkill-ttdist.sh' 2>/dev/null || true
setsid /home/keyspark/fastkill-ttdist.sh >/dev/null 2>&1 < /dev/null &
echo "fastkill-ttdist armed (pid $!)"

docker run -d --name ${NAME} --network host --ipc host --gpus all \
  --restart no \
  -v ${MODEL_DIR}:/model:ro \
  -v /home/keyspark/tt-dist:/work:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e GLOO_SOCKET_IFNAME=${IFACE} \
  --entrypoint python3 \
  ${IMG} \
  /work/twotower_dist.py --role ${ROLE} --model /model --master ${MASTER} "$@"

echo "launched ${NAME}; logs: docker logs -f ${NAME}"
