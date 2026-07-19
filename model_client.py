"""
model_client.py — GitOracle LLM Backend Connector
==================================================
Backend-agnostic LLM client factory.

Supported backends:
  - ollama  : Local Ollama via its OpenAI-compatible /v1 endpoint
  - groq    : Groq cloud API (fast inference, free tier available)

Both return a client whose .chat.completions.create() signature is identical,
so the rest of the codebase never needs to know which backend is active.
"""

import sys
import time
import threading
from config import CFG
from logger import get_logger

log = get_logger("ModelClient")


class ClientState:
    def __init__(self, api_key: str, client_index: int):
        from groq import Groq
        self.api_key = api_key
        # Mask the key for log/metrics safety: e.g. gsk_a1...b2
        masked = api_key
        if len(api_key) > 8:
            masked = f"{api_key[:6]}...{api_key[-4:]}"
        self.label = f"Key {client_index} ({masked})"
        self.client = Groq(api_key=api_key)
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.cooldown_until = 0.0  # monotonic time
        self.disabled = False
        self.disabled_reason = None
        self.last_used = 0.0  # monotonic time
        
        # Latency metrics
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0
        
        # Debugging metrics
        self.last_failure_reason = None
        self.last_failure_time = 0.0  # monotonic time

    def is_available(self, now: float) -> bool:
        if self.disabled:
            return False
        return now >= self.cooldown_until


class LLMLoadBalancer:
    def __init__(self, api_keys: str, model_name: str):
        self.model_name = model_name
        self.lock = threading.Lock()
        self.pointer = -1
        self.total_failovers = 0
        
        keys_list = [k.strip() for k in api_keys.split(",") if k.strip()]
        self.states = [ClientState(key, idx + 1) for idx, key in enumerate(keys_list)]
        
        # Mimic SDK completions structure
        class CompletionsProxy:
            def __init__(self, lb):
                self.lb = lb
            def create(self, *args, **kwargs):
                return self.lb.execute_with_failover("chat.completions.create", *args, **kwargs)

        class ChatProxy:
            def __init__(self, lb):
                self.completions = CompletionsProxy(lb)

        self.chat = ChatProxy(self)

    def get_metrics(self) -> dict:
        now_mono = time.monotonic()
        total_requests = sum(s.request_count for s in self.states)
        active_clients = sum(1 for s in self.states if not s.disabled)
        clients_in_cooldown = sum(1 for s in self.states if not s.disabled and now_mono < s.cooldown_until)
        
        keys_metrics = []
        for s in self.states:
            avg_latency = (s.total_latency_ms / s.success_count) if s.success_count > 0 else 0.0
            cooldown_rem = max(0.0, s.cooldown_until - now_mono)
            keys_metrics.append({
                "label": s.label,
                "requests": s.request_count,
                "successes": s.success_count,
                "failures": s.failure_count,
                "success_rate_percent": round((s.success_count / s.request_count * 100) if s.request_count > 0 else 100.0, 2),
                "cooldown_remaining_seconds": round(cooldown_rem, 1),
                "available": s.is_available(now_mono),
                "disabled": s.disabled,
                "disabled_reason": s.disabled_reason,
                "last_failure_reason": s.last_failure_reason,
                "last_failure_time_ago": round(now_mono - s.last_failure_time, 1) if s.last_failure_time > 0 else None,
                "avg_latency_ms": round(avg_latency, 1),
                "last_latency_ms": round(s.last_latency_ms, 1) if s.last_latency_ms > 0 else None,
            })
            
        return {
            "backend": "groq",
            "pool": {
                "total_requests": total_requests,
                "total_failovers": self.total_failovers,
                "active_clients": active_clients,
                "clients_in_cooldown": clients_in_cooldown
            },
            "keys": keys_metrics
        }

    def _select_client_state(self) -> ClientState:
        now = time.monotonic()
        n = len(self.states)
        if n == 0:
            raise ValueError("No LLM clients initialized in the load balancer.")
            
        # Check if all clients are cooling down or disabled
        disabled_count = sum(1 for s in self.states if s.disabled)
        cooldown_states = [s for s in self.states if not s.disabled and now < s.cooldown_until]
        
        if disabled_count == len(self.states):
            raise ValueError("All LLM clients in the load balancer have been permanently disabled due to authentication errors.")
            
        if len(cooldown_states) == (len(self.states) - disabled_count):
            min_cooldown = min(s.cooldown_until for s in cooldown_states)
            retry_after = max(1, int(min_cooldown - now))
            raise ValueError(
                f"All LLM clients are currently unavailable. "
                f"{len(cooldown_states)} client(s) cooling down. "
                f"{disabled_count} client(s) disabled. "
                f"Retry after approximately {retry_after} seconds."
            )
            
        # Round robin search for available client
        for _ in range(n):
            self.pointer = (self.pointer + 1) % n
            state = self.states[self.pointer]
            if state.is_available(now):
                state.last_used = now
                return state
                
        # Fallback (should not be reached if conditions above are correct)
        active_states = [s for s in self.states if not s.disabled]
        state = min(active_states, key=lambda s: s.cooldown_until)
        state.last_used = now
        return state

    def execute_with_failover(self, method_path, *args, **kwargs):
        # Determine retry budget based on number of active (non-disabled) clients
        active_count = sum(1 for s in self.states if not s.disabled)
        if active_count == 0:
            raise ValueError("All LLM API clients have been permanently disabled due to authentication errors.")
            
        for attempt in range(active_count):
            # Select key inside the thread lock ONLY
            with self.lock:
                state = self._select_client_state()
                # Record attempted request immediately upon selection
                state.request_count += 1
                
            log.info(f"Routing request to LLM load balancer: {state.label} (Attempt {attempt+1}/{active_count})")
            
            # Start timer
            start_time = time.perf_counter()
            
            try:
                # Dynamically resolve client.chat.completions.create
                parts = method_path.split(".")
                obj = state.client
                for part in parts:
                    obj = getattr(obj, part)
                
                # Execute completion call WITHOUT holding the lock
                result = obj(*args, **kwargs)
                
                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                state.success_count += 1
                state.last_latency_ms = latency_ms
                state.total_latency_ms += latency_ms
                
                return result
                
            except Exception as e:
                state.failure_count += 1
                
                status_code = getattr(e, "status_code", None)
                cooldown_dur = 0.0
                perm_disable = False
                reason = "Unknown Error"
                
                # Analyze failure type
                if status_code == 429 or "rate_limit" in type(e).__name__ or "RateLimit" in type(e).__name__ or "rate limit" in str(e).lower():
                    cooldown_dur = 60.0
                    reason = "429 Rate Limit Exceeded"
                elif status_code in (401, 403) or "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
                    perm_disable = True
                    reason = f"Authentication Failed (HTTP {status_code})"
                elif status_code is not None and 500 <= status_code <= 599:
                    cooldown_dur = 20.0
                    reason = f"HTTP {status_code} Internal Server Error"
                elif "timeout" in str(e).lower():
                    cooldown_dur = 15.0
                    reason = "Connection Timeout"
                else:
                    cooldown_dur = 15.0
                    reason = f"Transient Error: {type(e).__name__}"
                    
                now_mono = time.monotonic()
                state.last_failure_reason = reason
                state.last_failure_time = now_mono
                
                if perm_disable:
                    state.disabled = True
                    state.disabled_reason = reason
                    log.error(f"Critical error on {state.label}: {reason}. Permanently disabling this key.")
                else:
                    state.cooldown_until = now_mono + cooldown_dur
                    log.warning(
                        f"Transient error on {state.label}: {reason}. "
                        f"Placing key on cooldown for {cooldown_dur}s and failing over..."
                    )
                    
                if attempt < active_count - 1:
                    self.total_failovers += 1
                    continue
                
                # Final attempt failure
                raise e


def get_client_and_model():
    """
    Returns (client, model_name, backend_name).

    The client implements the OpenAI chat completions interface regardless
    of which backend is chosen — swap backends by changing .env only.
    """
    if CFG.llm_backend == "groq":
        try:
            from groq import Groq
        except ImportError:
            log.error("groq package not installed. Run: pip install groq>=0.8.0")
            sys.exit(1)
        client = LLMLoadBalancer(CFG.groq_api_key, CFG.groq_model)
        log.info(f"Backend: Groq | Model: {CFG.groq_model} | Loaded {len(client.states)} API key(s) in load balancer")
        return client, CFG.groq_model, "groq"

    # Default: Ollama via OpenAI-compatible endpoint
    # api_key="ollama" is a required-but-unchecked placeholder.
    from openai import OpenAI
    client = OpenAI(base_url=CFG.ollama_base_url, api_key="ollama")
    log.info(f"Backend: Ollama @ {CFG.ollama_base_url} | Model: {CFG.ollama_model}")
    return client, CFG.ollama_model, "ollama"


def check_connection(client, model: str, backend: str) -> bool:
    """
    Pre-flight connectivity check. Sends a minimal request and reports
    a clear, backend-specific action if it fails.

    Returns True on success, False on failure.
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        log.success(f"Connected to {backend} ({model})")
        return True
    except Exception as e:
        if backend == "ollama":
            log.error(f"Cannot reach Ollama at {CFG.ollama_base_url}")
            log.error(f"  1) Start Ollama:       ollama serve")
            log.error(f"  2) Pull the model:     ollama pull {model}")
        else:
            log.error(f"Cannot reach Groq. Check GROQ_API_KEY in your .env file.")
            log.error(f"  Underlying error: {e}")
        return False
