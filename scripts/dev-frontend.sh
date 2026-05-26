#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
cp -n .env.example .env || true
npm install
npm run dev
