#!/usr/bin/env bash
# Остаток попыток и история сдач (/me — попытку не тратит).
# Ключ: SUBMIT_KEY в .env или переменная окружения. Запуск из любого места:
#   bash scripts/check_remaining_submissions.sh
set -euo pipefail
cd "$(dirname "$0")/.."

read_env() { grep -E "^$1=" .env 2>/dev/null | head -n1 | cut -d= -f2- || true; }

KEY="${SUBMIT_KEY:-$(read_env SUBMIT_KEY)}"
EMAIL="${STUDENT_EMAIL:-$(read_env STUDENT_EMAIL)}"
if [ -z "$KEY" ] || [ -z "$EMAIL" ]; then
  echo "!! нужны SUBMIT_KEY и STUDENT_EMAIL (в .env или переменных окружения)"; exit 1
fi

curl -s --max-time 30 http://51.250.34.2:9000/me \
  -H "X-Student-Email: $EMAIL" \
  -H "X-Submit-Key: $KEY" \
  | python3 -m json.tool --no-ensure-ascii
