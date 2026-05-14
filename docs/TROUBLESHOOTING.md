# Troubleshooting

## Common Errors

### 403 Forbidden

**Cause:** Token expired or invalid credentials.

**Fix:**
1. Check token expiry: `python3 -c "import json; d=json.load(open('/path/to/kiro-auth-token.json')); print(d.get('expiresAt'))"`
2. Re-login via Kiro IDE or run `kiro-cli login`
3. Restart the gateway after token refresh

### 429 Too Many Requests

**Cause:** Rate limit or capacity exhaustion (`INSUFFICIENT_MODEL_CAPACITY`).

**Fix:**
1. The gateway automatically retries with backoff — wait a moment
2. If persistent, switch to a different model (e.g., Sonnet instead of Opus)
3. With multi-account setup, the gateway will failover to another account automatically

### 504 Gateway Timeout / Connection Reset

**Cause:** Network issue or AWS Q API timeout.

**Fix:**
1. Check your internet connection
2. If behind a firewall, configure `VPN_PROXY_URL` in `.env`
3. Try a different `KIRO_API_REGION` (default: `us-east-1`)

### "Improperly formed request" (400)

**Cause:** Request format issue, often from unsupported features.

**Fix:**
1. Enable debug logging (see below) to capture the full request
2. Check if you're using beta features not supported by the gateway
3. Verify your model name is valid (run `curl http://localhost:8765/v1/models`)

### "Model not found" (404)

**Cause:** Model name not recognized.

**Fix:**
1. List available models: `curl -H "x-api-key: YOUR_KEY" http://localhost:8765/v1/models`
2. Use the exact model ID from the list
3. The gateway normalizes names: `claude-sonnet-4-5`, `claude-sonnet-4.5`, and `claude-sonnet-4-5-20250929` all work

## Enabling Debug Logging

Set `DEBUG_MODE` in `.env`:

```bash
# Log only errors (default)
DEBUG_MODE=errors

# Log all requests (overwrites on each request)
DEBUG_MODE=all

# Log all requests in timestamped directories (corpus collection)
DEBUG_MODE=rotate
```

Debug logs are written to `debug_logs/` in the gateway directory.

## Checking Token Expiration

```bash
python3 -c "
import json, datetime, os
f = os.path.expanduser('~/.aws/sso/cache/kiro-auth-token.json')
d = json.load(open(f))
exp = datetime.datetime.fromisoformat(d['expiresAt'].replace('Z', '+00:00'))
now = datetime.datetime.now(datetime.timezone.utc)
mins = int((exp - now).total_seconds() / 60)
print(f'Token expires in {mins} minutes' if mins > 0 else f'Token EXPIRED {-mins} minutes ago')
"
```

## Verifying Gateway is Working

```bash
# Health check
curl http://localhost:8765/health

# List models
curl -H "x-api-key: YOUR_PROXY_API_KEY" http://localhost:8765/v1/models

# Test request
curl -X POST http://localhost:8765/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4.6", "max_tokens": 100, "messages": [{"role": "user", "content": "Say hello"}]}'
```

## Gateway Logs

```bash
# View live logs (systemd)
journalctl --user -u kiro-gateway -f

# View recent errors
journalctl --user -u kiro-gateway --since "1 hour ago" | grep ERROR
```

## Multi-Account Issues

If failover isn't working:
1. Check `credentials.json` has valid paths to token files
2. Verify each token file exists and is not expired
3. Check gateway logs for circuit breaker state changes
4. Set `CIRCUIT_BREAKER_RESET_TIMEOUT=60` to reset faster during testing
