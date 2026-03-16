# Evaluation Wiring, gRPC Monitor, Bus SubmitEvaluation — Design Spec

## Overview

Three independent enhancements that complete the human-in-the-loop evaluation loop and upgrade the monitor to use gRPC streaming.

## Item 1: Wire EvaluationGate into Coordinator

**Trigger:** `spec.evaluation_type == "human"` (already parsed from `Type: human` in workload spec's `## Evaluation` section).

**Behavior:** When evaluation is human, the coordinator does NOT use `result.primary_metric` from the executor. Instead, after each successful experiment completes in `_run_iteration()` Phase 3, it:

1. Publishes an `evaluation_requested` event with experiment details (experiment_id, dimension, params, any output paths)
2. Calls `eval_gate.request_evaluation(experiment_id, details)` — blocks the coordinator thread
3. Receives `(score, note)` back from the gate
4. If score is None (skipped/timeout), marks the experiment as failed
5. Otherwise, sets `result.primary_metric = score`

**Wiring path:**
- Coordinator.__init__ gains `eval_gate: EvaluationGate | None = None`
- CLI creates EvaluationGate when `--tui` AND `spec.evaluation_type == "human"`
- CLI passes eval_gate to Coordinator, ViewManager passes it through to ChaosApp
- If human eval needed but no TUI active (no eval_gate), log a warning and skip the experiment

**No TUI fallback:** Human evaluation requires the TUI modal. Without `--tui`, the coordinator logs a warning and treats the experiment as failed. This is intentional — stdin is occupied by the pause controller.

## Item 2: MonitorClient gRPC Subscribe Stream

**Replace HTTP polling with gRPC streaming.** The current `MonitorClient._poll_loop()` calls `/events?offset=N` which the bus doesn't actually serve — it only has `/publish`, `/commands`, `/healthz`, and Connect gRPC.

**Approach:** Use the Connect protocol's HTTP API directly (no `grpcio` dependency). The bus uses `connectrpc.com/connect` which supports Connect protocol over HTTP/1.1 for server streaming. The wire format is:
- POST to `/chaos.v1.BusService/Subscribe`
- Request body: protobuf-encoded `SubscribeRequest`
- Response: stream of length-prefixed protobuf `Event` messages

**However**, the simpler approach: add a `/events` SSE endpoint to the bus (Go side) that returns buffered + live events as newline-delimited JSON. This keeps the Python client dependency-free and matches the existing HTTP-based architecture.

**Chosen approach:** Add `/events` streaming endpoint to the bus. The Python MonitorClient reads it as a streaming HTTP response with chunked transfer encoding. Each line is a JSON event. Falls back to polling if streaming fails.

## Item 3: Bus-side SubmitEvaluation Handler

**Pattern:** Identical to PauseRun and ExtendBudget — marshal to JSON command, push to CommandQueue.

**Go side:** Add `SubmitEvaluation` method to `BusServer` in `server.go`. Regenerate Connect stubs from updated proto.

**Python side:** Coordinator's `_poll_bus_commands()` handles new `submit_evaluation` command type → calls `eval_gate.submit_evaluation(score, note)`.

This enables remote evaluation submission (e.g., from a web dashboard or CLI tool) in addition to the TUI modal.
