#!/bin/bash
# This script kills the inference server and the SSH tunnel.

SERVER_NAME=$1
LOCAL_PORT=$2
PRIVATE_KEY=$3
WORK_DIR=$4

if [ -z "${SERVER_NAME}" ] || [ -z "${LOCAL_PORT}" ] || [ -z "${PRIVATE_KEY}" ] || [ -z "${WORK_DIR}" ]; then
  echo "Usage: $0 <server_name> <local_port> <private_key> <WORK_DIR>"
  echo "Example: $0 user@host.com 5001 id_ed25519 /path/to/GenNarrate"
  exit 1
fi

echo -n "Killing remote python server on ${SERVER_NAME}... "
ssh "${SERVER_NAME}" "pkill -f 'python3 ${WORK_DIR}/inference/main.py'" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; }

sleep 1

echo -n "Killing local SSH tunnel on port ${LOCAL_PORT}... "
pkill -f "ssh -f -N -L ${LOCAL_PORT}:localhost:5000" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; exit 1; }
