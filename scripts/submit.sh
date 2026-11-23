#!/usr/bin/env bash
# Сдача ДЗ грейдеру. Ключ: SUBMIT_KEY в .env или переменная окружения.
# Запуск из корня проекта (make token читает .env):
#   bash scripts/submit.sh
# Адрес стенда переопределяется: BASE_URL=http://host:port bash scripts/submit.sh
# Ответ идёт несколько минут (грейдер прогоняет все критерии) — не прерывайте.
set -euo pipefail
cd "$(dirname "$0")/.."

read_env() { grep -E "^$1=" .env 2>/dev/null | head -n1 | cut -d= -f2- || true; }

KEY="${SUBMIT_KEY:-$(read_env SUBMIT_KEY)}"
if [ -z "$KEY" ]; then
  echo "!! задайте SUBMIT_KEY в .env или переменной окружения"; exit 1
fi
PORT="${GATEWAY_PORT:-$(read_env GATEWAY_PORT)}"; PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://$(hostname -I | awk '{print $1}'):${PORT}}"

echo ">> base_url: $BASE_URL"
curl -s --max-time 600 -X POST http://51.250.34.2:9000/submit \
  -H "Content-Type: application/json" \
  -H "X-Submit-Key: $KEY" \
  -d "{\"base_url\": \"$BASE_URL\", \"token\": \"$(make -s token)\"}" \
  | python3 -m json.tool --no-ensure-ascii
