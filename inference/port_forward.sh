#!/bin/bash
# This script starts up the inference server with port 
# forwarding on specified local port to server port 5000.

SERVER_NAME=$1
LOCAL_PORT=$2
PRIVATE_KEY=$3
WORK_DIR=$4

if [ -z "${SERVER_NAME}" ] || [ -z "${LOCAL_PORT}" ] || [ -z "${PRIVATE_KEY}" ] || [ -z "${WORK_DIR}" ]; then
  echo "Usage: $0 <server_name> <local_port> <private_key> <WORK_DIR>"
  echo "Example: $0 user@host.com 5001 id_ed25519 /path/to/GenNarrate"
  exit 1
fi

echo -n "Running python server on ${SERVER_NAME}..."
ssh -f "${SERVER_NAME}" "\
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate inference && \
cd ${WORK_DIR}/inference && \
setsid python3 main.py &" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; exit 1; }

sleep 1

echo -n "Forwarding port ${LOCAL_PORT} to ${SERVER_NAME}:5000..."
ssh -f -N -L "${LOCAL_PORT}:localhost:5000" -i ~/.ssh/${PRIVATE_KEY} "${SERVER_NAME}" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; exit 1; }

echo -e "To kill server and forwarding, run \033[0;34mkill_forwarding.sh\033[0m"