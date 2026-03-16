# Evaluation Wiring + gRPC Monitor + Bus SubmitEvaluation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the human-in-the-loop evaluation loop by wiring EvaluationGate into the coordinator, upgrading MonitorClient to use streaming events, and adding SubmitEvaluation to the bus.

**Architecture:** Three independent streams: (1) Python coordinator + CLI wiring for eval_gate, (2) Go bus `/events` SSE endpoint + Python MonitorClient rewrite, (3) Go bus SubmitEvaluation RPC + Python command handler. Each can be implemented and tested independently.

**Tech Stack:** Python 3.10+, Textual, Go 1.24, connectrpc, protobuf

---

## Task 1: Wire EvaluationGate into Coordinator

**Files:**
- Modify: `chaosengineer/core/coordinator.py` (add eval_gate param, modify _run_iteration Phase 3)
- Modify: `chaosengineer/cli.py` (create eval_gate, pass through)
- Modify: `chaosengineer/tui/views.py` (pass eval_gate to ChaosApp)
- Test: `tests/test_coordinator_eval.py` (new)

### Step 1: Write failing test — coordinator calls eval_gate for human evaluation

- [ ] Create `tests/test_coordinator_eval.py`:

```python
"""Tests for coordinator human-in-the-loop evaluation wiring."""
from unittest.mock import MagicMock, patch
import threading

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig, DimensionSpec, DimensionType
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(evaluation_type="automatic"):
    return WorkloadSpec(
        name="test",
        context="test workload",
        dimensions=[DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.01)],
        execution_command="echo test",
        evaluation_type=evaluation_type,
        primary_metric="accuracy",
        metric_direction="higher",
        metric_parse_command="echo 0.5",
        budget=BudgetConfig(max_experiments=5),
    )


def _make_coordinator(spec, eval_gate=None):
    dm = MagicMock()
    dm.pick_next_dimension.return_value = DimensionPlan(
        dimension_name="lr",
        values=[{"lr": 0.1}],
    )
    dm.discover_diverse_options = MagicMock(return_value=[])

    executor = MagicMock()
    result = MagicMock()
    result.primary_metric = 0.5
    result.cost_usd = 0.10
    result.error_message = None
    result.commit_hash = None
    result.to_dict.return_value = {"primary_metric": 0.5}
    executor.run_experiments.return_value = [result]

    logger = MagicMock()
    budget = BudgetTracker(spec.budget)
    baseline = Baseline(commit="HEAD", metric_value=0.0, metric_name="accuracy")

    return Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget,
        initial_baseline=baseline,
        eval_gate=eval_gate,
    )


class TestCoordinatorEvalGate:
    def test_human_eval_calls_eval_gate(self):
        """When evaluation_type=human and eval_gate is set, coordinator calls request_evaluation."""
        spec = _make_spec(evaluation_type="human")
        gate = EvaluationGate()

        # Pre-submit evaluation so the gate doesn't block forever
        def submit_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.submit_evaluation(0.85, "looks good")

        t = threading.Thread(target=submit_later, daemon=True)
        t.start()

        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()
        t.join(timeout=5)

        # The experiment should have used the human score
        assert coord.best_baseline.metric_value == 0.85

    def test_automatic_eval_does_not_call_gate(self):
        """When evaluation_type=automatic, eval_gate is not used even if provided."""
        spec = _make_spec(evaluation_type="automatic")
        gate = EvaluationGate()
        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()

        # Should use the executor's metric, not block on gate
        assert not gate.evaluation_needed.is_set()

    def test_human_eval_skip_marks_failed(self):
        """When human skips evaluation (score=None), experiment is treated as failed."""
        spec = _make_spec(evaluation_type="human")
        gate = EvaluationGate()

        def skip_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.skip_evaluation()

        t = threading.Thread(target=skip_later, daemon=True)
        t.start()

        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()
        t.join(timeout=5)

        # Skipped = no improvement from baseline of 0.0
        assert coord.best_baseline.metric_value == 0.0

    def test_human_eval_no_gate_logs_warning(self):
        """When evaluation_type=human but no eval_gate, experiment is skipped with warning."""
        spec = _make_spec(evaluation_type="human")
        coord = _make_coordinator(spec, eval_gate=None)
        coord.run()

        # Should complete without blocking, baseline unchanged
        assert coord.best_baseline.metric_value == 0.0
```

- [ ] Run test to verify it fails:

```bash
python -m pytest tests/test_coordinator_eval.py -v --tb=short
```

Expected: FAIL — Coordinator doesn't accept `eval_gate` param yet.

### Step 2: Implement coordinator eval_gate wiring

- [ ] Modify `chaosengineer/core/coordinator.py`:

Add `eval_gate` parameter to `__init__`:

```python
def __init__(
    self,
    ...
    pause_gate: "PauseGate | None" = None,
    eval_gate: "EvaluationGate | None" = None,
):
    ...
    self._eval_gate = eval_gate
```

Add `_request_human_evaluation` method:

```python
def _request_human_evaluation(self, exp: Experiment, result: ExperimentResult) -> float | None:
    """Block for human evaluation score. Returns score or None if skipped."""
    if self._eval_gate is None:
        self._log(Event(
            event="evaluation_skipped",
            data={
                "experiment_id": exp.experiment_id,
                "reason": "no_eval_gate",
            },
        ))
        return None

    details = {
        "experiment_id": exp.experiment_id,
        "dimension": exp.dimension,
        "params": exp.params,
    }
    self._log(Event(
        event="evaluation_requested",
        data=details,
    ))

    score, note = self._eval_gate.request_evaluation(exp.experiment_id, details)

    if score is not None:
        self._log(Event(
            event="evaluation_submitted",
            data={
                "experiment_id": exp.experiment_id,
                "score": score,
                "note": note,
            },
        ))
    return score
```

Modify `_run_iteration()` Phase 3 — in the success branch (the `else` after `if result.error_message`), add human eval override:

```python
else:
    # Human evaluation override
    if self.spec.evaluation_type == "human":
        score = self._request_human_evaluation(exp, result)
        if score is None:
            fail_experiment(exp, result)
            self._log(Event(
                event="worker_failed",
                data={
                    "experiment_id": exp.experiment_id,
                    "error": "human_evaluation_skipped",
                    "dimension": exp.dimension,
                    "params": exp.params,
                    "cost_usd": result.cost_usd,
                },
            ))
            release_worker(worker)
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd)
            results.append((exp, result))
            continue
        result.primary_metric = score

    complete_experiment(exp, result)
    self._log(Event(
        event="worker_completed",
        ...
    ))
```

- [ ] Run tests to verify they pass:

```bash
python -m pytest tests/test_coordinator_eval.py -v --tb=short
```

Expected: 4 PASS

### Step 3: Wire eval_gate through CLI and ViewManager

- [ ] Modify `chaosengineer/cli.py` in `_execute_run()`:

After creating `pause_gate` (line ~313), add:

```python
eval_gate = None
if spec.evaluation_type == "human":
    from chaosengineer.tui.eval_gate import EvaluationGate
    eval_gate = EvaluationGate()
```

Pass to Coordinator:

```python
coordinator = Coordinator(
    ...
    eval_gate=eval_gate,
)
```

Pass to ViewManager → ChaosApp. In the TUI block, update ViewManager usage:

```python
view_manager = ViewManager(bridge, pause_gate, pause_controller,
                            coordinator, status_display, eval_gate=eval_gate)
```

- [ ] Modify `chaosengineer/tui/views.py` — add `eval_gate` parameter to ViewManager.__init__:

```python
def __init__(
    self,
    bridge, pause_gate, pause_controller, coordinator, status_display,
    eval_gate=None,
):
    ...
    self._eval_gate = eval_gate
```

In `_enter_tui()`, pass eval_gate to ChaosApp:

```python
app = ChaosApp(
    bridge=self._bridge,
    pause_gate=self._pause_gate,
    coordinator=self._coordinator,
    pause_controller=self._pause_controller,
    eval_gate=self._eval_gate,
)
```

- [ ] Do the same for `_execute_resume()` in cli.py — same pattern.

- [ ] Run full TUI test suite:

```bash
python -m pytest tests/test_coordinator_eval.py tests/test_tui_app.py tests/test_view_manager.py tests/test_tui_integration.py -v --tb=short
```

Expected: All pass (no regressions).

### Step 4: Commit

- [ ] ```bash
git add chaosengineer/core/coordinator.py chaosengineer/cli.py chaosengineer/tui/views.py tests/test_coordinator_eval.py
git commit -m "feat: wire EvaluationGate into coordinator for human-in-the-loop evaluation"
```

---

## Task 2: Bus-side SubmitEvaluation Handler + Events Streaming Endpoint

**Files:**
- Modify: `bus/internal/server.go` (add SubmitEvaluation method)
- Create: `bus/internal/events.go` (SSE /events endpoint)
- Modify: `bus/internal/broker.go` (add Subscribe method returning channel for HTTP streaming)
- Modify: `bus/main.go` (register /events route)
- Test: `bus/internal/integration_test.go` (add SubmitEvaluation + events stream tests)

### Step 1: Write failing test — SubmitEvaluation queues command

- [ ] Add to `bus/internal/integration_test.go`:

```go
func TestIntegration_SubmitEvaluation(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	postEvent(baseURL, map[string]any{
		"event":  "run_started",
		"run_id": "run-eval",
	})

	client := chaosv1connect.NewBusServiceClient(
		http.DefaultClient,
		baseURL,
	)
	_, err := client.SubmitEvaluation(context.Background(), connect.NewRequest(
		&chaosv1.SubmitEvaluationRequest{
			RunId:        "run-eval",
			ExperimentId: "exp-0-0",
			Score:        0.85,
			Note:         "looks good",
		},
	))
	if err != nil {
		t.Fatal(err)
	}

	cmds, _ := getCommands(baseURL)
	if len(cmds) != 1 {
		t.Fatalf("got %d commands, want 1", len(cmds))
	}
	if cmds[0]["command"] != "submit_evaluation" {
		t.Errorf("got command %v", cmds[0]["command"])
	}
	if cmds[0]["experiment_id"] != "exp-0-0" {
		t.Errorf("got experiment_id %v", cmds[0]["experiment_id"])
	}
	if cmds[0]["score"] != 0.85 {
		t.Errorf("got score %v", cmds[0]["score"])
	}
}
```

- [ ] Run test to verify it fails:

```bash
cd bus && go test ./internal/ -run TestIntegration_SubmitEvaluation -v
```

Expected: FAIL — SubmitEvaluation method not implemented.

### Step 2: Regenerate Go Connect stubs from updated proto

- [ ] The proto was already updated with `SubmitEvaluation`. Regenerate stubs:

```bash
cd bus && protoc \
  --go_out=gen --go_opt=paths=source_relative \
  --connect-go_out=gen --connect-go_opt=paths=source_relative \
  -I ../proto \
  chaos/v1/events.proto chaos/v1/bus.proto
```

If `protoc` or plugins aren't available, manually add the method to the generated files following the exact pattern of ExtendBudget. The key additions in `bus/gen/chaos/v1/chaosv1connect/bus.connect.go`:
- Add `BusServiceSubmitEvaluationProcedure` constant
- Add `SubmitEvaluation` to the client/handler interfaces and implementations

And in `bus/gen/chaos/v1/bus.pb.go`:
- Add `SubmitEvaluationRequest` and `SubmitEvaluationResponse` message types

### Step 3: Implement SubmitEvaluation in server.go

- [ ] Add to `bus/internal/server.go`:

```go
// SubmitEvaluation queues an evaluation result for the coordinator.
func (s *BusServer) SubmitEvaluation(
	ctx context.Context,
	req *connect.Request[chaosv1.SubmitEvaluationRequest],
) (*connect.Response[chaosv1.SubmitEvaluationResponse], error) {
	runID := req.Msg.RunId
	if runID == "" {
		runID = s.Queue.CurrentRun()
	}
	cmd, _ := json.Marshal(map[string]any{
		"command":       "submit_evaluation",
		"run_id":        runID,
		"experiment_id": req.Msg.ExperimentId,
		"score":         req.Msg.Score,
		"note":          req.Msg.Note,
	})
	s.Queue.Push(cmd)
	return connect.NewResponse(&chaosv1.SubmitEvaluationResponse{}), nil
}
```

- [ ] Run test:

```bash
cd bus && go test ./internal/ -run TestIntegration_SubmitEvaluation -v
```

Expected: PASS

### Step 4: Write failing test — /events streaming endpoint

- [ ] Add to `bus/internal/integration_test.go`:

```go
func TestIntegration_EventsStream(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	// Publish an event first
	postEvent(baseURL, map[string]any{
		"event":  "run_started",
		"run_id": "run-stream",
	})

	// Connect to /events stream
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("got status %d", resp.StatusCode)
	}

	// Read first line — should be the replayed event
	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	line := strings.TrimSpace(string(buf[:n]))

	var event map[string]any
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("invalid JSON: %v (raw: %s)", err, line)
	}
	if event["event"] != "run_started" {
		t.Errorf("got event %v, want run_started", event["event"])
	}
}
```

- [ ] Run test to verify it fails:

```bash
cd bus && go test ./internal/ -run TestIntegration_EventsStream -v
```

Expected: FAIL — 404 /events not found.

### Step 5: Implement /events streaming endpoint

- [ ] Create `bus/internal/events.go`:

```go
// bus/internal/events.go
package internal

import (
	"encoding/json"
	"net/http"

	"google.golang.org/protobuf/encoding/protojson"
)

// NewEventsHandler returns an HTTP handler for GET /events.
// It replays buffered events and then streams live events as newline-delimited JSON.
// The connection stays open until the client disconnects.
func NewEventsHandler(broker *Broker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		runID := r.URL.Query().Get("run_id")

		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		ch, cancel := broker.Subscribe(runID)
		defer cancel()

		marshaler := protojson.MarshalOptions{EmitUnpopulated: false}

		for {
			select {
			case <-r.Context().Done():
				return
			case event, ok := <-ch:
				if !ok {
					return
				}
				// Convert protobuf Event to flat JSON matching Python's event format
				data := map[string]any{
					"event":  event.EventType,
					"run_id": event.RunId,
				}
				if event.Ts != nil {
					data["ts"] = event.Ts.AsTime().Format("2006-01-02T15:04:05Z")
				}
				if event.Data != nil {
					// Merge data fields into top level
					jsonBytes, _ := marshaler.Marshal(event.Data)
					var fields map[string]any
					json.Unmarshal(jsonBytes, &fields)
					for k, v := range fields {
						data[k] = v
					}
				}

				line, _ := json.Marshal(data)
				w.Write(line)
				w.Write([]byte("\n"))
				flusher.Flush()
			}
		}
	}
}
```

- [ ] Register in `bus/main.go` — add after the `/commands` handler:

```go
mux.HandleFunc("/events", internal.NewEventsHandler(broker))
```

- [ ] Run test:

```bash
cd bus && go test ./internal/ -run TestIntegration_EventsStream -v
```

Expected: PASS

### Step 6: Run full Go test suite

- [ ] ```bash
cd bus && go test ./internal/ -v
```

Expected: All pass.

### Step 7: Commit

- [ ] ```bash
git add bus/internal/server.go bus/internal/events.go bus/main.go bus/internal/integration_test.go bus/gen/
git commit -m "feat(bus): add SubmitEvaluation RPC + /events streaming endpoint"
```

---

## Task 3: Upgrade MonitorClient to Use Streaming + Coordinator Command Handler

**Files:**
- Modify: `chaosengineer/tui/monitor.py` (replace HTTP polling with /events stream)
- Modify: `chaosengineer/core/coordinator.py` (handle submit_evaluation command)
- Modify: `tests/test_monitor.py` (update for streaming)
- Create: `tests/test_coordinator_eval_bus.py` (test command handler)

### Step 1: Write failing test — MonitorClient uses streaming

- [ ] Update `tests/test_monitor.py` — replace the HTTP polling mock with a streaming mock:

```python
"""Tests for MonitorClient and readonly TUI mode."""
from unittest.mock import MagicMock, patch, PropertyMock
import io
import json
import threading
import time

import pytest

from chaosengineer.tui.monitor import MonitorClient
from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


class FakeStreamResponse:
    """Simulates a streaming HTTP response with newline-delimited JSON."""

    def __init__(self, events):
        lines = [json.dumps(e) + "\n" for e in events]
        self._data = "".join(lines).encode()
        self._stream = io.BytesIO(self._data)
        self.status = 200

    def read(self, amt=-1):
        return self._stream.read(amt)

    def readline(self):
        return self._stream.readline()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestMonitorClientStreaming:
    def test_streams_events_to_bridge(self):
        """MonitorClient should read streaming /events and publish to bridge."""
        events = [
            {"event": "run_started", "run_id": "r1"},
            {"event": "worker_completed", "run_id": "r1", "metric": 0.5},
        ]
        fake_resp = FakeStreamResponse(events)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            client = MonitorClient(bus_url="http://fake:1234")
            client.start()
            time.sleep(0.3)
            client.stop()

        snapshot = client.bridge.snapshot()
        assert len(snapshot) == 2
        assert snapshot[0]["event"] == "run_started"
        assert snapshot[1]["event"] == "worker_completed"

    def test_reconnects_on_error(self):
        """MonitorClient should reconnect after stream errors."""
        call_count = 0
        events = [{"event": "run_started", "run_id": "r1"}]

        def fake_urlopen(req, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            return FakeStreamResponse(events)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client = MonitorClient(bus_url="http://fake:1234")
            client.start()
            time.sleep(1.5)  # enough for retry
            client.stop()

        assert call_count >= 2

    def test_run_id_filter(self):
        """MonitorClient passes run_id to the URL."""
        captured_urls = []

        def capture_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            captured_urls.append(url)
            raise ConnectionError("stop")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            client = MonitorClient(bus_url="http://fake:1234", run_id="run-abc")
            client.start()
            time.sleep(0.3)
            client.stop()

        assert any("run_id=run-abc" in u for u in captured_urls)
```

- [ ] Run test to verify it fails:

```bash
python -m pytest tests/test_monitor.py::TestMonitorClientStreaming -v --tb=short
```

Expected: FAIL — MonitorClient still uses old polling format.

### Step 2: Rewrite MonitorClient to use streaming

- [ ] Replace `chaosengineer/tui/monitor.py`:

```python
"""Monitor client -- connects to chaos-bus /events stream and feeds to local EventBridge."""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

from chaosengineer.tui.bridge import EventBridge

logger = logging.getLogger(__name__)


class MonitorClient:
    """Connects to a remote chaos-bus ``/events`` streaming endpoint.

    The bus returns newline-delimited JSON (NDJSON) over a long-lived HTTP
    connection. MonitorClient reads lines in a background thread and publishes
    each event to a local :class:`EventBridge` so the TUI can display them.
    """

    def __init__(self, bus_url: str, run_id: str = ""):
        self._bus_url = bus_url.rstrip("/")
        self._run_id = run_id
        self.bridge = EventBridge()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background streaming thread."""
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the streaming thread to stop."""
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _stream_loop(self) -> None:
        """Connect to /events and read newline-delimited JSON events."""
        while not self._stop.is_set():
            try:
                url = f"{self._bus_url}/events"
                if self._run_id:
                    url += f"?run_id={self._run_id}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    for line_bytes in resp:
                        if self._stop.is_set():
                            break
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            self.bridge.publish(event)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON line: %s", line)
            except Exception:
                logger.debug("Monitor stream failed, reconnecting...", exc_info=True)
            # Wait before reconnecting
            self._stop.wait(timeout=1.0)
```

- [ ] Run streaming tests:

```bash
python -m pytest tests/test_monitor.py::TestMonitorClientStreaming -v --tb=short
```

Expected: PASS

### Step 3: Update existing readonly tests

- [ ] The `TestReadonlyMode` tests in `tests/test_monitor.py` don't depend on MonitorClient internals — they test ChaosApp's readonly flag. Verify they still pass:

```bash
python -m pytest tests/test_monitor.py -v --tb=short
```

Expected: All pass. Remove any old `TestMonitorClient` class that tests the polling approach and replace with the new `TestMonitorClientStreaming`.

### Step 4: Write failing test — coordinator handles submit_evaluation command

- [ ] Create `tests/test_coordinator_eval_bus.py`:

```python
"""Tests for coordinator handling submit_evaluation bus command."""
from unittest.mock import MagicMock
import threading

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.workloads.parser import WorkloadSpec


class TestSubmitEvaluationCommand:
    def test_poll_commands_handles_submit_evaluation(self):
        """_poll_bus_commands should call eval_gate.submit_evaluation on submit_evaluation command."""
        spec = WorkloadSpec(
            name="test", evaluation_type="human",
            primary_metric="score", metric_direction="higher",
            budget=BudgetConfig(max_experiments=5),
        )
        gate = EvaluationGate()
        gate.evaluation_needed.set()  # Simulate waiting for evaluation

        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "submit_evaluation", "experiment_id": "exp-0-0", "score": 0.75, "note": "ok"},
        ]

        coord = Coordinator(
            spec=spec,
            decision_maker=MagicMock(),
            executor=MagicMock(),
            logger=logger,
            budget=MagicMock(),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )
        coord._poll_bus_commands()

        assert gate.score == 0.75
        assert gate.note == "ok"
        assert gate.evaluation_ready.is_set()

    def test_poll_commands_ignores_eval_without_gate(self):
        """submit_evaluation command is ignored when no eval_gate is set."""
        spec = WorkloadSpec(name="test", budget=BudgetConfig(max_experiments=5))
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "submit_evaluation", "score": 0.75},
        ]

        coord = Coordinator(
            spec=spec,
            decision_maker=MagicMock(),
            executor=MagicMock(),
            logger=logger,
            budget=MagicMock(),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
        )
        # Should not raise
        coord._poll_bus_commands()
```

- [ ] Run test to verify it fails:

```bash
python -m pytest tests/test_coordinator_eval_bus.py -v --tb=short
```

Expected: FAIL — _poll_bus_commands doesn't handle submit_evaluation.

### Step 5: Implement submit_evaluation command handler

- [ ] Modify `chaosengineer/core/coordinator.py` `_poll_bus_commands()`:

```python
def _poll_bus_commands(self) -> None:
    """Poll message bus for remote commands (pause, extend_budget, submit_evaluation)."""
    if not hasattr(self.logger, "poll_commands"):
        return
    for cmd in self.logger.poll_commands():
        if cmd.get("command") == "pause" and self._pause_controller:
            self._pause_controller.pause_requested = True
        elif cmd.get("command") == "extend_budget":
            self.extend_budget(
                add_cost=cmd.get("add_cost_usd", 0),
                add_experiments=cmd.get("add_experiments", 0),
                add_time=cmd.get("add_time_seconds", 0),
            )
        elif cmd.get("command") == "submit_evaluation" and self._eval_gate:
            score = cmd.get("score")
            note = cmd.get("note", "")
            if score is not None:
                self._eval_gate.submit_evaluation(float(score), note)
            else:
                self._eval_gate.skip_evaluation()
```

- [ ] Run test:

```bash
python -m pytest tests/test_coordinator_eval_bus.py -v --tb=short
```

Expected: PASS

### Step 6: Run full test suite

- [ ] ```bash
python -m pytest tests/test_monitor.py tests/test_coordinator_eval.py tests/test_coordinator_eval_bus.py tests/test_tui_app.py tests/test_view_manager.py tests/test_tui_integration.py -v --tb=short
```

Expected: All pass.

### Step 7: Commit

- [ ] ```bash
git add chaosengineer/tui/monitor.py chaosengineer/core/coordinator.py tests/test_monitor.py tests/test_coordinator_eval_bus.py
git commit -m "feat: streaming MonitorClient + submit_evaluation command handler"
```

---

## Parallelism Notes

**Task 1** and **Task 2** are fully independent — different files, different languages.

**Task 3** depends on Task 1 (eval_gate on coordinator) and Task 2 (/events endpoint). However:
- Step 4-5 of Task 3 (submit_evaluation command handler) only needs the `eval_gate` field from Task 1
- Steps 1-3 of Task 3 (MonitorClient rewrite) only needs the `/events` endpoint from Task 2

**Recommended parallel dispatch:**
- Agent A: Task 1 (Python coordinator eval wiring)
- Agent B: Task 2 (Go bus SubmitEvaluation + /events endpoint)
- Agent C: Task 3 (MonitorClient rewrite + command handler) — after A and B complete, OR in parallel if the agent handles merge conflicts
