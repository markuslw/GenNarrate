#!/bin/bash
# This script kills the inference server and the SSH tunnel.

SERVER_NAME=$1
LOCAL_PORT=$2
PRIVATE_KEY=$3
FILE_PATH=$4

if [ -z "${SERVER_NAME}" ] || [ -z "${LOCAL_PORT}" ] || [ -z "${PRIVATE_KEY}" ] || [ -z "${FILE_PATH}" ]; then
  echo "Usage: $0 <server_name> <local_port> <private_key> <file_path>"
  echo "Example: $0 <user@host.com> <3001 or 5001> <id_rsa or id_ed25519> /path/to/main.py"
  exit 1
fi

echo -n "Killing remote python server on ${SERVER_NAME}... "
ssh "${SERVER_NAME}" "pkill -f 'python3 ${FILE_PATH}'" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; }

sleep 1

echo -n "Killing local SSH tunnel on port ${LOCAL_PORT}... "
pkill -f "ssh -f -N -L ${LOCAL_PORT}:localhost:5000" && \
echo -e "\033[0;32mOK\033[0m" || { echo -e "\033[0;31mFAILED\033[0m"; exit 1; }
