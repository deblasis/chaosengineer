# Message Bus Design

## Goal

Add a generic event bus between the ChaosEngineer coordinator and its consumers (TUI, web dashboard, JSONL file writer). The bus is the source of truth for events — `events.jsonl` becomes a persistence subscriber rather than the primary data store. The bus also carries commands in the reverse direction (pause, evaluate, extend budget) from UIs back to the coordinator.

## Architecture

A Go binary (`chaos-bus`) acts as the event bus. The Python coordinator publishes events to it via HTTP POST (JSON). Consumers subscribe via Connect gRPC server-streaming. Commands flow back via gRPC RPCs, queued in the bus and polled by the coordinator at existing pause check points. The bus is auto-spawned by the CLI as a subprocess and shuts down after the run ends.

```
Coordinator (Python)
    │
    │  HTTP POST /publish (JSON)
    ▼
┌──────────────────────────────────┐
│         Event Bus (Go)           │
│                                  │
│  • Accept JSON, wrap in protobuf │
│  • Fan out to subscribers        │
│  • In-memory replay buffer       │
│  • Command queue (GET /commands) │
│  • Built-in JSONL file writer    │
└──────────┬───────────────────────┘
           │  Connect gRPC streams
           ├──→ TUI subscriber
           ├──→ Web dashboard subscriber
           ├──→ Future subscribers
           │
           │  Internal goroutine
           └──→ JSONL file writer (events.jsonl)

Commands (reverse):
  TUI/Web → gRPC RPC → Bus queue → Coordinator polls → PauseController/BudgetTracker
```

## Protobuf Schema

```protobuf
// proto/chaos/v1/events.proto

syntax = "proto3";
package chaos.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

message Event {
  string event_type = 1;
  google.protobuf.Timestamp ts = 2;
  string run_id = 3;
  google.protobuf.Struct data = 4;
}
```

```protobuf
// proto/chaos/v1/bus.proto

syntax = "proto3";
package chaos.v1;

import "chaos/v1/events.proto";

service BusService {
  rpc Subscribe(SubscribeRequest) returns (stream Event);
  rpc PauseRun(PauseRunRequest) returns (PauseRunResponse);
  rpc ExtendBudget(ExtendBudgetRequest) returns (ExtendBudgetResponse);
}

message SubscribeRequest {
  string run_id = 1;  // empty = current run
}

message PauseRunRequest { string run_id = 1; }
message PauseRunResponse {}

message ExtendBudgetRequest {
  string run_id = 1;
  double add_cost_usd = 2;
  int32 add_experiments = 3;
  double add_time_seconds = 4;
}
message ExtendBudgetResponse {}
```

Uses `google.protobuf.Struct` for event data rather than typed messages per event type. This keeps the schema stable — new event types (added on the Python side) don't require proto changes or recompilation. Consumers switch on `event_type` string and interpret the `data` Struct. The trade-off is weaker compile-time type safety, which is acceptable for UI consumers that mostly display data.

`ResumeRun` and `SubmitEvaluation` RPCs are deferred to future specs (TUI and human-in-the-loop respectively). Only `PauseRun` and `ExtendBudget` are implemented in this spec. The proto schema can be extended when those features are designed.

## Components

### Bus Server (Go)

**File:** `bus/main.go`, `bus/internal/`

A single Go binary with three subsystems:

**Publish endpoint** (`bus/internal/publisher.go`):
- `POST /publish` — accepts JSON, extracts `event` → `event_type` and `run_id`, puts remaining fields into `data` Struct. Adds server-side timestamp if not provided.
- Local-only (binds to `127.0.0.1` by default, configurable via `--host` flag for remote).
- Returns 200 immediately. Fire-and-forget from the coordinator's perspective.

**Broker** (`bus/internal/broker.go`):
- Maintains an `[]Event` replay buffer per run, reset on `run_started`.
- Maintains a set of subscriber channels.
- On new event: append to buffer, send to all subscriber channels.
- `Subscribe()` implementation: send all buffered events, then switch to live channel. Seamless from the client's perspective.
- Thread-safe via mutex on the buffer and subscriber set.

**Command queue** (`bus/internal/commands.go`):
- Maintains a queue of pending commands per run.
- gRPC RPCs (`PauseRun`, `ExtendBudget`) push to the queue.
- `GET /commands?run_id=X` returns and drains the queue as a JSON array.
- Thread-safe via mutex.
- The bus infers the current `run_id` from the first `run_started` event. Subscribers with an empty `run_id` receive events for the current run. Commands with an empty `run_id` target the current run. If no `run_started` has been received yet, subscribers block until one arrives.

**Command JSON format:** Each command in the array has a `command` field and command-specific fields:
```json
[
  {"command": "pause", "run_id": "run-abc12345"},
  {"command": "extend_budget", "run_id": "run-abc12345", "add_cost_usd": 10.0, "add_experiments": 5, "add_time_seconds": 0}
]
```

**File writer** (`bus/internal/filewriter.go`):
- A goroutine that subscribes to the broker like any other subscriber.
- Appends each event as a flat JSON line to the output file (same format as current `EventLogger`).
- Fields: `ts` (ISO 8601), `event` (event type), plus all data fields merged at top level.
- If the write fails (disk full, file deleted), logs a warning and continues. Does not crash the bus or block other subscribers.
- Enabled via `--output-file` flag.

**Format fidelity:** The file writer must produce the exact same flat JSON format as the current `EventLogger`. The protobuf `Struct` type preserves JSON structure through the bus, and the file writer serializes it back to flat JSON. Concrete round-trip example:

Input (HTTP POST to `/publish`):
```json
{"event": "worker_completed", "run_id": "run-abc", "experiment_id": "exp-0-0", "dimension": "lr", "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.12}
```

Output (line in `events.jsonl`):
```json
{"ts": "2026-03-16T14:23:45.123456+00:00", "event": "worker_completed", "run_id": "run-abc", "experiment_id": "exp-0-0", "dimension": "lr", "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.12}
```

The bus adds `ts` if not present. All other fields pass through unchanged. The `data` Struct is flattened back to top-level keys — no nesting introduced. Note: proto3 Struct represents all numbers as `double`. Since `events.jsonl` is read by Python's `json.loads()` which handles float/int transparently, this is not a problem.

**Lifecycle:**
- Accepts flags: `--port` (0 = auto-assign), `--host` (default `127.0.0.1`), `--output-file` (optional), `--shutdown-delay` (default 30s).
- Prints `{"port": N}` to stdout on startup so the CLI can discover the actual port.
- Exposes `GET /healthz` — returns 200 with `{"status": "ok"}`. The CLI polls this after reading the port to confirm the bus is ready before proceeding.
- On receiving SIGTERM or SIGINT: continues accepting events during the shutdown delay (so the coordinator's final `run_completed`/`run_paused` events are not lost), then stops accepting and exits. The CLI sends SIGTERM *after* the coordinator's `run()` returns, so the final events are already published before shutdown begins.

### EventPublisher (Python)

**File:** `chaosengineer/metrics/publisher.py`

Replaces `EventLogger` as the coordinator's event sink. Same `log(Event)` interface — the coordinator doesn't change.

**Methods:**
- `__init__(bus_url: str | None, fallback_path: Path | None = None)` — if `bus_url` is provided, tries to reach the bus via `GET /healthz`. If unreachable (or `bus_url` is None) and `fallback_path` is provided, creates an `EventLogger` as fallback. If neither works, raises.
- `log(event: Event) -> None` — POST JSON to `bus_url/publish`. On connection error, falls back to `EventLogger` if available, otherwise logs warning to stderr.
- `poll_commands() -> list[dict]` — GET `bus_url/commands`. Returns empty list on error or when in fallback mode. The coordinator calls this at existing pause check points.
- `read_events(event_type: str | None = None) -> list[dict]` — always reads from the JSONL file at `fallback_path`. When the bus is running, the bus writes this file via its built-in file writer, so `read_events` reads what the bus wrote. When in fallback mode, `EventPublisher` writes the file directly via its internal `EventLogger`, so `read_events` reads what the fallback wrote. Either way, the file is the read source. This means `build_snapshot()` and `ScenarioRunner` work unchanged.

**No new dependencies.** Uses `urllib.request` from stdlib for HTTP. JSON serialization via `json` module.

**Fallback behavior:** If the bus is not reachable at init time, `EventPublisher` transparently delegates all calls to `EventLogger`. This means:
- All existing tests work without a Go binary
- CI doesn't need Go installed for Python tests
- Users without the bus binary get the same experience as today

### Coordinator Changes

**File:** `chaosengineer/core/coordinator.py`

Minimal changes:
- `_run_loop()` — at each existing pause check point, also call `self.logger.poll_commands()` (if the method exists — duck-typed check via `hasattr`). For each command, dispatch to the appropriate handler:
  - `"pause"` → `self._pause_controller.pause_requested = True` (reuses existing pause infrastructure)
  - `"extend_budget"` → mutates `self.budget.config` in-place:
    ```python
    if cmd.get("add_cost_usd"):
        self.budget.config = BudgetConfig(
            max_api_cost=(self.budget.config.max_api_cost or 0) + cmd["add_cost_usd"],
            max_experiments=self.budget.config.max_experiments,
            max_wall_time_seconds=self.budget.config.max_wall_time_seconds,
            max_plateau_iterations=self.budget.config.max_plateau_iterations,
        )
    # Same pattern for add_experiments, add_time_seconds
    ```
    This mirrors the existing budget extension pattern in `_execute_resume()` (cli.py lines 314-334) where `BudgetConfig` objects are rebuilt with updated values. No new method on `BudgetTracker` is needed.

The coordinator continues to use `self.logger.log(event)` for publishing. It does not know whether the logger is an `EventPublisher` or `EventLogger`.

**File ownership:** When the bus is running, exactly one writer owns `events.jsonl` — the bus's built-in file writer goroutine. The `EventPublisher` does NOT write to the file; it only posts to the bus. The `EventPublisher`'s internal `EventLogger` fallback is only activated when the bus is unreachable, in which case the bus file writer is also not running. There is never a dual-writer situation.

### CLI Changes

**File:** `chaosengineer/cli.py`

In `_execute_run()` and `_execute_resume()`:
1. Locate the bus binary (env var → repo path → PATH).
2. If found, spawn as subprocess: `chaos-bus --port 0 --output-file <events.jsonl>`.
3. Read port from stdout.
4. Create `EventPublisher(bus_url=f"http://localhost:{port}", fallback_path=log_path)`.
5. If bus binary not found, create `EventPublisher(bus_url=None, fallback_path=log_path)` which falls back to `EventLogger` immediately.
6. On run end (after `coordinator.run()` returns), send SIGTERM to bus subprocess. The coordinator's final events (`run_completed`/`run_paused`) are already published at this point. Don't wait — the bus handles its own shutdown delay for subscriber catch-up.

## Package Structure

```
autoresearch/
├── chaosengineer/              # Python package (existing)
│   ├── core/
│   │   └── coordinator.py      # Modified: poll_commands at check points
│   ├── metrics/
│   │   ├── logger.py           # Unchanged (used as fallback)
│   │   └── publisher.py        # NEW: EventPublisher
│   └── cli.py                  # Modified: spawn bus, use publisher
│
├── proto/                       # NEW: shared protobuf definitions
│   └── chaos/v1/
│       ├── events.proto
│       └── bus.proto
│
├── bus/                         # NEW: Go module
│   ├── go.mod
│   ├── go.sum
│   ├── main.go
│   ├── internal/
│   │   ├── server.go           # Connect gRPC service
│   │   ├── broker.go           # Replay buffer + fan-out
│   │   ├── publisher.go        # HTTP POST /publish
│   │   ├── commands.go         # Command queue + GET /commands
│   │   └── filewriter.go       # JSONL subscriber goroutine
│   └── gen/                     # Generated protobuf/connect code
│       └── chaos/v1/
│
├── buf.yaml                     # NEW: buf schema configuration
└── buf.gen.yaml                 # NEW: codegen (Go + future connect-web)
```

**Binary discovery order:**
1. `CHAOS_BUS_BIN` environment variable
2. `bus/chaos-bus` relative to the repo root (development builds)
3. `chaos-bus` on system PATH

## Compatibility

### Backward Compatibility

- **EventLogger unchanged** — existing tests and `build_snapshot()` work as before.
- **events.jsonl format unchanged** — same flat JSON lines. Only the writer changes (bus goroutine vs. Python `EventLogger`).
- **Coordinator interface unchanged** — still calls `self.logger.log(event)`. Duck-typed, no ABC change needed.
- **Resume works unchanged** — `build_snapshot()` reads `events.jsonl` which the bus writes in the same format.
- **No new Python dependencies** — stdlib `urllib` for HTTP.

### Graceful Degradation

The bus is optional. If the binary is not installed:
- `EventPublisher` falls back to `EventLogger`
- No status line or TUI changes
- `events.jsonl` written directly as today
- `poll_commands()` returns empty list (no remote pause/commands)
- All existing functionality preserved

## Testing

### Go Tests (`bus/`)

- `broker_test.go` — replay buffer: new subscriber gets all buffered events, then live events. Buffer resets on `run_started`. Concurrent subscriber add/remove.
- `publisher_test.go` — JSON parsing: valid events, malformed input, missing fields. Timestamp injection when not provided.
- `commands_test.go` — queue/drain: commands accumulate, poll returns and clears. Concurrent RPC + poll.
- `filewriter_test.go` — output format matches `EventLogger` format exactly. Survives write errors without crashing.
- `integration_test.go` — start bus, POST events, subscribe via gRPC, verify ordered delivery with replay.

### Python Tests

- `test_publisher.py` — mock HTTP server: verify POST format, verify JSON structure. Verify fallback to EventLogger when bus unreachable. Verify `poll_commands()` returns commands and handles errors.
- `test_commands.py` — verify `poll_commands` result integrates with PauseController flags (pause command → `pause_requested = True`).
- Existing coordinator tests unchanged (use EventLogger directly, no bus).

### Integration Test

One end-to-end test (skipped if bus binary not built):
1. Build bus binary
2. Start bus on random port
3. Run coordinator with EventPublisher pointed at bus
4. Subscribe via gRPC, verify events arrive
5. Send PauseRun RPC, verify coordinator pauses
6. Verify `events.jsonl` contains same events
