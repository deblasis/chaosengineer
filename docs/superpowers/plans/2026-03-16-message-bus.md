# Message Bus Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Go event bus between the ChaosEngineer coordinator and its consumers (TUI, web, JSONL file writer), with the bus as the source of truth and graceful fallback to direct file writes when the bus is unavailable.

**Architecture:** A Go binary (`chaos-bus`) serves as the event bus. The Python coordinator publishes events via HTTP POST JSON. Consumers subscribe via Connect gRPC streaming. Commands flow back via gRPC RPCs, polled by the coordinator at existing pause checkpoints. The bus is auto-spawned by the CLI and includes a built-in JSONL file writer.

**Tech Stack:** Go 1.23+, Connect gRPC (`connectrpc.com/connect`), Protocol Buffers, buf for codegen, Python stdlib `urllib` for HTTP

**Spec:** `docs/superpowers/specs/2026-03-16-message-bus-design.md`

---

## Prerequisites

- Go 1.23+ (`go version`)
- buf 1.65+ (`buf --version`)
- Python 3.10+ with pytest

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `proto/chaos/v1/events.proto` | Event protobuf message |
| `proto/chaos/v1/bus.proto` | BusService gRPC definition |
| `buf.yaml` | buf module configuration |
| `buf.gen.yaml` | buf codegen configuration |
| `bus/go.mod` | Go module definition |
| `bus/main.go` | Binary entry point with CLI flags and lifecycle |
| `bus/internal/broker.go` | Replay buffer + fan-out to subscribers |
| `bus/internal/broker_test.go` | Broker unit tests |
| `bus/internal/publisher.go` | HTTP POST `/publish` handler |
| `bus/internal/publisher_test.go` | Publisher unit tests |
| `bus/internal/commands.go` | Command queue + GET `/commands` handler |
| `bus/internal/commands_test.go` | Command queue unit tests |
| `bus/internal/filewriter.go` | JSONL subscriber goroutine |
| `bus/internal/filewriter_test.go` | File writer unit tests |
| `bus/internal/server.go` | Connect gRPC service implementation |
| `bus/internal/integration_test.go` | Full server integration test |
| `chaosengineer/metrics/publisher.py` | EventPublisher (bus + fallback) |
| `tests/test_publisher.py` | EventPublisher unit tests |
| `tests/test_bus_commands.py` | Coordinator command polling tests |

### Modified Files

| File | Changes |
|------|---------|
| `chaosengineer/core/coordinator.py` | Add `_poll_bus_commands()` helper, call at pause checkpoints |
| `chaosengineer/cli.py` | Spawn bus subprocess, use EventPublisher, add `_find_bus_binary()` and `_wait_for_bus()` |
| `.gitignore` | Add `bus/chaos-bus`, `bus/gen/` |

---

## Chunk 1: Foundation

### Task 0: Protobuf Schema Files

**Context:** These proto files define the event and service contract shared between Go and future consumers. Python never touches protobuf directly — it posts JSON and the Go bus converts internally.

**Files:**
- Create: `proto/chaos/v1/events.proto`
- Create: `proto/chaos/v1/bus.proto`

- [ ] **Step 1: Create events.proto**

```protobuf
// proto/chaos/v1/events.proto
syntax = "proto3";
package chaos.v1;

option go_package = "chaos-bus/gen/chaos/v1;chaosv1";

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

message Event {
  string event_type = 1;
  google.protobuf.Timestamp ts = 2;
  string run_id = 3;
  google.protobuf.Struct data = 4;
}
```

- [ ] **Step 2: Create bus.proto**

```protobuf
// proto/chaos/v1/bus.proto
syntax = "proto3";
package chaos.v1;

option go_package = "chaos-bus/gen/chaos/v1;chaosv1";

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

- [ ] **Step 3: Verify proto files are valid**

Run: `buf lint proto/`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add proto/
git commit -m "feat(bus): add protobuf schema for events and bus service"
```

### Task 1: Buf Config + Go Module + Code Generation

**Context:** buf generates Go protobuf and Connect gRPC code from the proto files. The Go module lives in `bus/` with module name `chaos-bus`. Generated code goes to `bus/gen/`.

**Files:**
- Create: `buf.yaml`
- Create: `buf.gen.yaml`
- Create: `bus/go.mod`
- Generated: `bus/gen/chaos/v1/*.pb.go`, `bus/gen/chaos/v1/chaosv1connect/*.connect.go`

- [ ] **Step 1: Create buf.yaml**

```yaml
# buf.yaml
version: v2
modules:
  - path: proto
```

- [ ] **Step 2: Create buf.gen.yaml**

```yaml
# buf.gen.yaml
version: v2
plugins:
  - remote: buf.build/protocolbuffers/go
    out: bus/gen
    opt: paths=source_relative
  - remote: buf.build/connectrpc/go
    out: bus/gen
    opt: paths=source_relative
```

- [ ] **Step 3: Create Go module**

```bash
mkdir -p bus && cd bus && go mod init chaos-bus && cd ..
```

- [ ] **Step 4: Run buf generate**

```bash
buf generate
```

Expected: Files created in `bus/gen/chaos/v1/`:
- `events.pb.go` (protobuf types)
- `bus.pb.go` (protobuf types)
- `chaosv1connect/bus.connect.go` (Connect gRPC service stubs)

- [ ] **Step 5: Add generated code dependencies**

```bash
cd bus && go mod tidy && cd ..
```

- [ ] **Step 6: Verify Go module compiles**

```bash
cd bus && go build ./... && cd ..
```

Expected: No errors (nothing to build yet, but imports resolve)

- [ ] **Step 7: Update .gitignore**

Add to `.gitignore`:
```
bus/chaos-bus
```

Note: `bus/gen/` is committed (standard Go practice — allows building without buf).

- [ ] **Step 8: Commit**

```bash
git add buf.yaml buf.gen.yaml bus/go.mod bus/go.sum bus/gen/ .gitignore
git commit -m "feat(bus): add buf config, Go module, and generated protobuf code"
```

---

## Chunk 2: Go Bus Core

### Task 2: Broker

**Context:** The broker is the core data structure — an in-memory replay buffer that fans out events to subscribers. It resets its buffer on `run_started`. New subscribers get all buffered events (replay) then switch to live delivery. Thread-safe via `sync.RWMutex`.

**Files:**
- Create: `bus/internal/broker.go`
- Test: `bus/internal/broker_test.go`

- [ ] **Step 1: Write the failing tests**

```go
// bus/internal/broker_test.go
package internal

import (
	"testing"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func makeEvent(eventType, runID string) *chaosv1.Event {
	return &chaosv1.Event{
		EventType: eventType,
		RunId:     runID,
		Ts:        timestamppb.Now(),
		Data:      &structpb.Struct{},
	}
}

func recvTimeout(ch <-chan *chaosv1.Event, timeout time.Duration) (*chaosv1.Event, bool) {
	select {
	case e := <-ch:
		return e, true
	case <-time.After(timeout):
		return nil, false
	}
}

func TestBroker_ReplayOnSubscribe(t *testing.T) {
	b := NewBroker()
	b.Publish(makeEvent("run_started", "run-1"))
	b.Publish(makeEvent("worker_completed", "run-1"))

	ch, cancel := b.Subscribe("")
	defer cancel()

	e1, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for first replayed event")
	}
	if e1.EventType != "run_started" {
		t.Errorf("got %s, want run_started", e1.EventType)
	}

	e2, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for second replayed event")
	}
	if e2.EventType != "worker_completed" {
		t.Errorf("got %s, want worker_completed", e2.EventType)
	}
}

func TestBroker_LiveDelivery(t *testing.T) {
	b := NewBroker()
	ch, cancel := b.Subscribe("")
	defer cancel()

	b.Publish(makeEvent("test_event", "run-1"))

	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("timeout waiting for live event")
	}
	if e.EventType != "test_event" {
		t.Errorf("got %s, want test_event", e.EventType)
	}
}

func TestBroker_BufferResetsOnRunStarted(t *testing.T) {
	b := NewBroker()
	b.Publish(makeEvent("old_event", "run-1"))
	b.Publish(makeEvent("run_started", "run-2"))
	b.Publish(makeEvent("new_event", "run-2"))

	ch, cancel := b.Subscribe("")
	defer cancel()

	// Should only get run_started and new_event (old_event was cleared)
	var types []string
	for i := 0; i < 2; i++ {
		e, ok := recvTimeout(ch, time.Second)
		if !ok {
			t.Fatalf("timeout at event %d", i)
		}
		types = append(types, e.EventType)
	}

	if types[0] != "run_started" || types[1] != "new_event" {
		t.Errorf("got %v, want [run_started, new_event]", types)
	}
}

func TestBroker_MultipleSubscribers(t *testing.T) {
	b := NewBroker()
	ch1, cancel1 := b.Subscribe("")
	defer cancel1()
	ch2, cancel2 := b.Subscribe("")
	defer cancel2()

	b.Publish(makeEvent("broadcast", "run-1"))

	for i, ch := range []<-chan *chaosv1.Event{ch1, ch2} {
		e, ok := recvTimeout(ch, time.Second)
		if !ok {
			t.Fatalf("subscriber %d: timeout", i)
		}
		if e.EventType != "broadcast" {
			t.Errorf("subscriber %d: got %s, want broadcast", i, e.EventType)
		}
	}
}

func TestBroker_CurrentRun(t *testing.T) {
	b := NewBroker()
	if b.CurrentRun() != "" {
		t.Errorf("expected empty, got %s", b.CurrentRun())
	}

	b.Publish(makeEvent("run_started", "run-abc"))
	if b.CurrentRun() != "run-abc" {
		t.Errorf("expected run-abc, got %s", b.CurrentRun())
	}
}

func TestBroker_CancelUnsubscribes(t *testing.T) {
	b := NewBroker()
	_, cancel := b.Subscribe("")
	cancel()

	// Publishing after cancel should not panic
	b.Publish(makeEvent("after_cancel", "run-1"))
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd bus && go test ./internal/ -v -run TestBroker`
Expected: FAIL — `NewBroker` undefined

- [ ] **Step 3: Implement the broker**

```go
// bus/internal/broker.go
package internal

import (
	"sync"

	chaosv1 "chaos-bus/gen/chaos/v1"
)

// Broker manages an in-memory replay buffer and fans out events to subscribers.
type Broker struct {
	mu          sync.RWMutex
	buffer      []*chaosv1.Event
	subscribers map[chan *chaosv1.Event]struct{}
	currentRun  string
}

func NewBroker() *Broker {
	return &Broker{
		subscribers: make(map[chan *chaosv1.Event]struct{}),
	}
}

// Publish sends an event to all subscribers and appends to the replay buffer.
// Resets the buffer on "run_started" events.
func (b *Broker) Publish(event *chaosv1.Event) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if event.EventType == "run_started" {
		b.buffer = b.buffer[:0]
		b.currentRun = event.RunId
	}
	b.buffer = append(b.buffer, event)

	for ch := range b.subscribers {
		select {
		case ch <- event:
		default:
			// Slow subscriber — drop event rather than block others
		}
	}
}

// Subscribe returns a channel that receives all buffered events (replay)
// followed by live events. Call the returned cancel function to unsubscribe.
func (b *Broker) Subscribe(runID string) (<-chan *chaosv1.Event, func()) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Size channel to fit replay + headroom for live events
	bufSize := len(b.buffer) + 256
	ch := make(chan *chaosv1.Event, bufSize)

	// Replay buffered events while holding lock (no concurrent publishes)
	for _, event := range b.buffer {
		if runID == "" || event.RunId == runID {
			ch <- event
		}
	}

	b.subscribers[ch] = struct{}{}

	cancel := func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		delete(b.subscribers, ch)
	}

	return ch, cancel
}

// CurrentRun returns the run_id from the most recent run_started event.
func (b *Broker) CurrentRun() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.currentRun
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd bus && go test ./internal/ -v -run TestBroker`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bus/internal/broker.go bus/internal/broker_test.go
git commit -m "feat(bus): add broker with replay buffer and fan-out"
```

### Task 3: Command Queue

**Context:** The command queue accepts commands from gRPC RPCs (PauseRun, ExtendBudget) and serves them to the coordinator via `GET /commands`. Commands are accumulated and drained atomically. Thread-safe via `sync.Mutex`. Implemented before the publisher because the publisher handler takes an optional `*CommandQueue` parameter.

**Files:**
- Create: `bus/internal/commands.go`
- Test: `bus/internal/commands_test.go`

- [ ] **Step 1: Write the failing tests**

```go
// bus/internal/commands_test.go
package internal

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCommandQueue_PushAndDrain(t *testing.T) {
	q := NewCommandQueue()

	cmd1, _ := json.Marshal(map[string]string{"command": "pause", "run_id": "run-1"})
	cmd2, _ := json.Marshal(map[string]any{"command": "extend_budget", "add_cost_usd": 10.0})

	q.Push(cmd1)
	q.Push(cmd2)

	cmds := q.Drain()
	if len(cmds) != 2 {
		t.Fatalf("got %d commands, want 2", len(cmds))
	}

	// Drain again — should be empty
	cmds2 := q.Drain()
	if len(cmds2) != 0 {
		t.Errorf("got %d commands after drain, want 0", len(cmds2))
	}
}

func TestCommandQueue_DrainEmpty(t *testing.T) {
	q := NewCommandQueue()
	cmds := q.Drain()
	if cmds != nil {
		t.Errorf("expected nil, got %v", cmds)
	}
}

func TestCommandQueue_SetCurrentRun(t *testing.T) {
	q := NewCommandQueue()
	q.SetCurrentRun("run-abc")
	if q.CurrentRun() != "run-abc" {
		t.Errorf("got %s, want run-abc", q.CurrentRun())
	}
}

func TestCommandsHandler_ReturnsDrainedCommands(t *testing.T) {
	q := NewCommandQueue()
	cmd, _ := json.Marshal(map[string]string{"command": "pause"})
	q.Push(cmd)

	handler := NewCommandsHandler(q)
	req := httptest.NewRequest(http.MethodGet, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}

	body, _ := io.ReadAll(w.Body)
	var cmds []json.RawMessage
	if err := json.Unmarshal(body, &cmds); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(cmds) != 1 {
		t.Errorf("got %d commands, want 1", len(cmds))
	}
}

func TestCommandsHandler_EmptyArray(t *testing.T) {
	q := NewCommandQueue()
	handler := NewCommandsHandler(q)

	req := httptest.NewRequest(http.MethodGet, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	body, _ := io.ReadAll(w.Body)
	var cmds []json.RawMessage
	json.Unmarshal(body, &cmds)
	if len(cmds) != 0 {
		t.Errorf("got %d commands, want 0", len(cmds))
	}
}

func TestCommandsHandler_MethodNotAllowed(t *testing.T) {
	q := NewCommandQueue()
	handler := NewCommandsHandler(q)

	req := httptest.NewRequest(http.MethodPost, "/commands", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want 405", w.Code)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd bus && go test ./internal/ -v -run "TestCommand"`
Expected: FAIL — `NewCommandQueue` undefined

- [ ] **Step 3: Implement the command queue**

```go
// bus/internal/commands.go
package internal

import (
	"encoding/json"
	"net/http"
	"sync"
)

// CommandQueue accumulates commands from gRPC RPCs and serves them
// to the coordinator via GET /commands.
type CommandQueue struct {
	mu         sync.Mutex
	queue      []json.RawMessage
	currentRun string
}

func NewCommandQueue() *CommandQueue {
	return &CommandQueue{}
}

func (q *CommandQueue) SetCurrentRun(runID string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.currentRun = runID
}

func (q *CommandQueue) CurrentRun() string {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.currentRun
}

// Push adds a command to the queue.
func (q *CommandQueue) Push(cmd json.RawMessage) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queue = append(q.queue, cmd)
}

// Drain returns all queued commands and clears the queue.
// Returns nil if empty.
func (q *CommandQueue) Drain() []json.RawMessage {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.queue) == 0 {
		return nil
	}
	cmds := q.queue
	q.queue = nil
	return cmds
}

// NewCommandsHandler returns an HTTP handler for GET /commands.
// Returns the drained queue as a JSON array.
func NewCommandsHandler(queue *CommandQueue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cmds := queue.Drain()
		if cmds == nil {
			cmds = []json.RawMessage{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(cmds)
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd bus && go test ./internal/ -v -run "TestCommand"`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bus/internal/commands.go bus/internal/commands_test.go
git commit -m "feat(bus): add command queue with GET /commands endpoint"
```

### Task 4: HTTP Publisher Endpoint

**Context:** The `/publish` endpoint accepts JSON from the Python coordinator, extracts `event` and `run_id` fields, converts remaining fields to a protobuf `Struct`, and publishes via the broker. Adds a server-side timestamp if not provided. Depends on `CommandQueue` type from Task 3.

**Files:**
- Create: `bus/internal/publisher.go`
- Test: `bus/internal/publisher_test.go`

- [ ] **Step 1: Write the failing tests**

```go
// bus/internal/publisher_test.go
package internal

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPublishHandler_ValidEvent(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	body := `{"event":"worker_completed","run_id":"run-abc","experiment_id":"exp-0","metric":2.5}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", w.Code)
	}

	// Verify event was published to broker
	ch, cancel := broker.Subscribe("")
	defer cancel()

	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("no event in broker")
	}
	if e.EventType != "worker_completed" {
		t.Errorf("event_type: got %s, want worker_completed", e.EventType)
	}
	if e.RunId != "run-abc" {
		t.Errorf("run_id: got %s, want run-abc", e.RunId)
	}
	if e.Ts == nil {
		t.Error("timestamp should be set")
	}

	// Data should contain remaining fields
	data := e.Data.AsMap()
	if data["experiment_id"] != "exp-0" {
		t.Errorf("experiment_id: got %v, want exp-0", data["experiment_id"])
	}
	if data["metric"] != 2.5 {
		t.Errorf("metric: got %v, want 2.5", data["metric"])
	}
}

func TestPublishHandler_TimestampPreserved(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	ts := "2026-03-16T14:00:00.000000+00:00"
	body := `{"event":"test","ts":"` + ts + `"}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	ch, cancel := broker.Subscribe("")
	defer cancel()
	e, ok := recvTimeout(ch, time.Second)
	if !ok {
		t.Fatal("no event")
	}

	got := e.Ts.AsTime()
	if got.Year() != 2026 || got.Month() != 3 || got.Day() != 16 {
		t.Errorf("timestamp not preserved: %v", got)
	}
}

func TestPublishHandler_MissingEventField(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	body := `{"run_id":"run-1","metric":1.0}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestPublishHandler_MalformedJSON(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString("{bad"))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want 400", w.Code)
	}
}

func TestPublishHandler_MethodNotAllowed(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	req := httptest.NewRequest(http.MethodGet, "/publish", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want 405", w.Code)
	}
}

func TestPublishHandler_UpdatesCommandQueueCurrentRun(t *testing.T) {
	broker := NewBroker()
	queue := NewCommandQueue()
	handler := NewPublishHandler(broker, queue)

	body := `{"event":"run_started","run_id":"run-xyz"}`
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if queue.CurrentRun() != "run-xyz" {
		t.Errorf("got %s, want run-xyz", queue.CurrentRun())
	}
}

func TestPublishHandler_NestedData(t *testing.T) {
	broker := NewBroker()
	handler := NewPublishHandler(broker, nil)

	payload := map[string]any{
		"event":  "worker_completed",
		"params": map[string]any{"lr": 0.01, "batch_size": 32},
		"metric": 2.5,
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/publish", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	ch, cancel := broker.Subscribe("")
	defer cancel()
	e, _ := recvTimeout(ch, time.Second)

	data := e.Data.AsMap()
	params, ok := data["params"].(map[string]any)
	if !ok {
		t.Fatalf("params not a map: %T", data["params"])
	}
	if params["lr"] != 0.01 {
		t.Errorf("lr: got %v, want 0.01", params["lr"])
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd bus && go test ./internal/ -v -run TestPublishHandler`
Expected: FAIL — `NewPublishHandler` undefined

- [ ] **Step 3: Implement the publisher handler**

```go
// bus/internal/publisher.go
package internal

import (
	"encoding/json"
	"net/http"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// NewPublishHandler returns an HTTP handler for POST /publish.
// It accepts JSON, extracts "event" and "run_id", converts remaining
// fields to a protobuf Struct, and publishes via the broker.
// If queue is non-nil, updates it on run_started events.
func NewPublishHandler(broker *Broker, queue *CommandQueue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}

		eventType, _ := raw["event"].(string)
		if eventType == "" {
			http.Error(w, "missing 'event' field", http.StatusBadRequest)
			return
		}

		runID, _ := raw["run_id"].(string)

		// Extract timestamp or generate one
		var ts time.Time
		if tsStr, ok := raw["ts"].(string); ok {
			parsed, err := time.Parse(time.RFC3339Nano, tsStr)
			if err == nil {
				ts = parsed
			} else {
				ts = time.Now().UTC()
			}
			delete(raw, "ts")
		} else {
			ts = time.Now().UTC()
		}

		// Remove extracted fields — rest goes into data Struct
		delete(raw, "event")
		delete(raw, "run_id")

		data, err := structpb.NewStruct(raw)
		if err != nil {
			http.Error(w, "invalid data fields", http.StatusBadRequest)
			return
		}

		event := &chaosv1.Event{
			EventType: eventType,
			RunId:     runID,
			Ts:        timestamppb.New(ts),
			Data:      data,
		}

		broker.Publish(event)

		if eventType == "run_started" && queue != nil {
			queue.SetCurrentRun(runID)
		}

		w.WriteHeader(http.StatusOK)
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd bus && go test ./internal/ -v -run TestPublishHandler`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bus/internal/publisher.go bus/internal/publisher_test.go
git commit -m "feat(bus): add HTTP POST /publish endpoint"
```

### Task 5: JSONL File Writer

**Context:** The file writer is a goroutine that subscribes to the broker and appends each event as a flat JSON line. Must produce the exact same format as Python's `EventLogger`: `{"ts": "2026-...", "event": "...", "run_id": "...", ...data fields...}`. Write failures are logged, not fatal.

**Important:** The timestamp format must match Python's `datetime.now(timezone.utc).isoformat()` output exactly: `2026-03-16T14:23:45.123456+00:00` (microsecond precision, `+00:00` timezone).

**Files:**
- Create: `bus/internal/filewriter.go`
- Test: `bus/internal/filewriter_test.go`

- [ ] **Step 1: Write the failing tests**

```go
// bus/internal/filewriter_test.go
package internal

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestFileWriter_OutputFormat(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	// Publish an event with known data
	ts := time.Date(2026, 3, 16, 14, 23, 45, 123456000, time.UTC)
	data, _ := structpb.NewStruct(map[string]any{
		"experiment_id": "exp-0-0",
		"dimension":     "lr",
		"params":        map[string]any{"lr": 0.01},
		"metric":        2.5,
		"cost_usd":      0.12,
	})
	event := &chaosv1.Event{
		EventType: "worker_completed",
		RunId:     "run-abc",
		Ts:        timestamppb.New(ts),
		Data:      data,
	}
	broker.Publish(event)

	// Give file writer time to flush
	time.Sleep(100 * time.Millisecond)

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}

	var record map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &record); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// Verify fields
	if record["event"] != "worker_completed" {
		t.Errorf("event: got %v", record["event"])
	}
	if record["run_id"] != "run-abc" {
		t.Errorf("run_id: got %v", record["run_id"])
	}
	if record["experiment_id"] != "exp-0-0" {
		t.Errorf("experiment_id: got %v", record["experiment_id"])
	}
	if record["metric"] != 2.5 {
		t.Errorf("metric: got %v", record["metric"])
	}

	// Verify timestamp format matches Python isoformat()
	tsStr, ok := record["ts"].(string)
	if !ok {
		t.Fatal("ts not a string")
	}
	if tsStr != "2026-03-16T14:23:45.123456+00:00" {
		t.Errorf("ts: got %s, want 2026-03-16T14:23:45.123456+00:00", tsStr)
	}

	// Verify nested data preserved
	params, ok := record["params"].(map[string]any)
	if !ok {
		t.Fatal("params not a map")
	}
	if params["lr"] != 0.01 {
		t.Errorf("params.lr: got %v", params["lr"])
	}
}

func TestFileWriter_NoRunID(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	event := &chaosv1.Event{
		EventType: "iteration_started",
		Ts:        timestamppb.Now(),
		Data:      &structpb.Struct{},
	}
	broker.Publish(event)
	time.Sleep(100 * time.Millisecond)

	content, _ := os.ReadFile(path)
	var record map[string]any
	json.Unmarshal([]byte(strings.TrimSpace(string(content))), &record)

	// run_id should be absent when empty
	if _, exists := record["run_id"]; exists {
		t.Error("run_id should not be present when empty")
	}
}

func TestFileWriter_MultipleEvents(t *testing.T) {
	broker := NewBroker()
	path := t.TempDir() + "/events.jsonl"

	cancel := StartFileWriter(broker, path)
	defer cancel()

	for i := 0; i < 5; i++ {
		broker.Publish(makeEvent("test", "run-1"))
	}
	time.Sleep(200 * time.Millisecond)

	content, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 5 {
		t.Errorf("got %d lines, want 5", len(lines))
	}
}

func TestFileWriter_SurvivesWriteError(t *testing.T) {
	broker := NewBroker()
	// Use a path that doesn't exist — file writer should log warning, not panic
	cancel := StartFileWriter(broker, "/nonexistent/dir/events.jsonl")
	defer cancel()

	// Publish should not panic
	broker.Publish(makeEvent("test", "run-1"))
	time.Sleep(100 * time.Millisecond)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd bus && go test ./internal/ -v -run TestFileWriter`
Expected: FAIL — `StartFileWriter` undefined

- [ ] **Step 3: Implement the file writer**

```go
// bus/internal/filewriter.go
package internal

import (
	"encoding/json"
	"log"
	"os"
	"sort"

	chaosv1 "chaos-bus/gen/chaos/v1"
)

// tsFormat matches Python's datetime.now(timezone.utc).isoformat() output.
const tsFormat = "2006-01-02T15:04:05.000000-07:00"

// StartFileWriter subscribes to the broker and writes events as flat JSONL.
// Returns a cancel function that stops the writer.
func StartFileWriter(broker *Broker, path string) func() {
	ch, cancel := broker.Subscribe("")

	go func() {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("filewriter: cannot open %s: %v", path, err)
			// Drain channel so broker doesn't block
			for range ch {
			}
			return
		}
		defer f.Close()

		for event := range ch {
			line := EventToJSONL(event)
			if _, err := f.WriteString(line + "\n"); err != nil {
				log.Printf("filewriter: write error: %v", err)
			}
		}
	}()

	return cancel
}

// EventToJSONL converts a protobuf Event to a flat JSON line matching
// the format produced by Python's EventLogger.
func EventToJSONL(event *chaosv1.Event) string {
	record := make(map[string]any)

	if event.Ts != nil {
		record["ts"] = event.Ts.AsTime().UTC().Format(tsFormat)
	}

	record["event"] = event.EventType

	if event.RunId != "" {
		record["run_id"] = event.RunId
	}

	if event.Data != nil {
		for k, v := range event.Data.AsMap() {
			record[k] = v
		}
	}

	b, _ := json.Marshal(sortedMap(record))
	return string(b)
}

// sortedMap returns an ordered representation for deterministic JSON output.
// Uses json.Marshal on a slice of key-value pairs with deterministic key order:
// ts first, event second, run_id third, then remaining keys alphabetically.
func sortedMap(m map[string]any) json.Marshaler {
	return orderedMap(m)
}

type orderedMap map[string]any

func (o orderedMap) MarshalJSON() ([]byte, error) {
	// Priority keys in order
	priority := []string{"ts", "event", "run_id"}
	var keys []string
	seen := make(map[string]bool)

	for _, k := range priority {
		if _, ok := o[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}

	var rest []string
	for k := range o {
		if !seen[k] {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	keys = append(keys, rest...)

	buf := []byte{'{'}
	for i, k := range keys {
		if i > 0 {
			buf = append(buf, ',')
		}
		keyJSON, _ := json.Marshal(k)
		valJSON, _ := json.Marshal(o[k])
		buf = append(buf, keyJSON...)
		buf = append(buf, ':')
		buf = append(buf, valJSON...)
	}
	buf = append(buf, '}')
	return buf, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd bus && go test ./internal/ -v -run TestFileWriter`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bus/internal/filewriter.go bus/internal/filewriter_test.go
git commit -m "feat(bus): add JSONL file writer with format fidelity"
```

---

## Chunk 3: Go Bus Server

### Task 6: Connect gRPC Server + Main Binary

**Context:** Wire all components together. `server.go` implements the Connect `BusServiceHandler` interface (Subscribe streaming, PauseRun, ExtendBudget). `main.go` parses flags, creates all components, sets up the HTTP mux, manages lifecycle (signal handling, shutdown delay).

**Key behaviors:**
- Prints `{"port": N}` to stdout on startup (CLI reads this)
- `GET /healthz` returns 200 with `{"status":"ok"}`
- On SIGTERM/SIGINT: waits shutdown-delay, then exits
- Logs go to stderr (stdout reserved for port JSON)

**Files:**
- Create: `bus/internal/server.go`
- Create: `bus/main.go`

- [ ] **Step 1: Create server.go (Connect service implementation)**

```go
// bus/internal/server.go
package internal

import (
	"context"
	"encoding/json"

	chaosv1 "chaos-bus/gen/chaos/v1"

	"connectrpc.com/connect"
)

// BusServer implements the BusService Connect gRPC interface.
type BusServer struct {
	Broker *Broker
	Queue  *CommandQueue
}

func NewBusServer(broker *Broker, queue *CommandQueue) *BusServer {
	return &BusServer{Broker: broker, Queue: queue}
}

// Subscribe streams events to the client. Replays buffered events first,
// then delivers live events until the client disconnects.
func (s *BusServer) Subscribe(
	ctx context.Context,
	req *connect.Request[chaosv1.SubscribeRequest],
	stream *connect.ServerStream[chaosv1.Event],
) error {
	runID := req.Msg.RunId
	ch, cancel := s.Broker.Subscribe(runID)
	defer cancel()

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-ch:
			if !ok {
				return nil
			}
			if err := stream.Send(event); err != nil {
				return err
			}
		}
	}
}

// PauseRun queues a pause command for the coordinator to pick up.
func (s *BusServer) PauseRun(
	ctx context.Context,
	req *connect.Request[chaosv1.PauseRunRequest],
) (*connect.Response[chaosv1.PauseRunResponse], error) {
	runID := req.Msg.RunId
	if runID == "" {
		runID = s.Queue.CurrentRun()
	}
	cmd, _ := json.Marshal(map[string]string{
		"command": "pause",
		"run_id":  runID,
	})
	s.Queue.Push(cmd)
	return connect.NewResponse(&chaosv1.PauseRunResponse{}), nil
}

// ExtendBudget queues a budget extension command for the coordinator.
func (s *BusServer) ExtendBudget(
	ctx context.Context,
	req *connect.Request[chaosv1.ExtendBudgetRequest],
) (*connect.Response[chaosv1.ExtendBudgetResponse], error) {
	runID := req.Msg.RunId
	if runID == "" {
		runID = s.Queue.CurrentRun()
	}
	cmd, _ := json.Marshal(map[string]any{
		"command":          "extend_budget",
		"run_id":           runID,
		"add_cost_usd":     req.Msg.AddCostUsd,
		"add_experiments":  req.Msg.AddExperiments,
		"add_time_seconds": req.Msg.AddTimeSeconds,
	})
	s.Queue.Push(cmd)
	return connect.NewResponse(&chaosv1.ExtendBudgetResponse{}), nil
}
```

- [ ] **Step 2: Create main.go**

```go
// bus/main.go
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"chaos-bus/gen/chaos/v1/chaosv1connect"
	"chaos-bus/internal"
)

func main() {
	port := flag.Int("port", 0, "listen port (0 = auto-assign)")
	host := flag.String("host", "127.0.0.1", "listen host")
	outputFile := flag.String("output-file", "", "JSONL output file path")
	shutdownDelay := flag.Duration("shutdown-delay", 30*time.Second, "delay before shutdown")
	flag.Parse()

	// Logs to stderr — stdout is reserved for the port JSON
	log.SetOutput(os.Stderr)

	broker := internal.NewBroker()
	queue := internal.NewCommandQueue()

	// Start file writer if configured
	if *outputFile != "" {
		cancel := internal.StartFileWriter(broker, *outputFile)
		defer cancel()
	}

	// HTTP mux: publish, commands, healthz, and Connect gRPC
	mux := http.NewServeMux()
	mux.HandleFunc("/publish", internal.NewPublishHandler(broker, queue))
	mux.HandleFunc("/commands", internal.NewCommandsHandler(queue))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	busServer := internal.NewBusServer(broker, queue)
	path, handler := chaosv1connect.NewBusServiceHandler(busServer)
	mux.Handle(path, handler)

	listener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", *host, *port))
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	actualPort := listener.Addr().(*net.TCPAddr).Port

	// Print port to stdout for CLI discovery
	json.NewEncoder(os.Stdout).Encode(map[string]int{"port": actualPort})

	server := &http.Server{Handler: mux}

	// Signal handling: wait shutdown-delay then exit
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		<-sigCh
		log.Printf("Received signal, shutting down in %s", *shutdownDelay)
		time.Sleep(*shutdownDelay)
		server.Shutdown(context.Background())
	}()

	log.Printf("chaos-bus listening on %s:%d", *host, actualPort)
	if err := server.Serve(listener); err != http.ErrServerClosed {
		log.Fatalf("serve: %v", err)
	}
}
```

- [ ] **Step 3: Build the binary**

```bash
cd bus && go build -o chaos-bus . && cd ..
```

Expected: Binary `bus/chaos-bus` created with no errors

- [ ] **Step 4: Smoke test the binary**

```bash
# Start bus, read port from stdout JSON, test healthz, then kill
bus/chaos-bus --port 0 --shutdown-delay 1s > /tmp/bus-port.json 2>/dev/null &
BUS_PID=$!
sleep 0.5
PORT=$(python3 -c "import json; print(json.load(open('/tmp/bus-port.json'))['port'])")
curl -s "http://127.0.0.1:${PORT}/healthz"
kill $BUS_PID 2>/dev/null; wait $BUS_PID 2>/dev/null
rm -f /tmp/bus-port.json
```

Expected: `{"status":"ok"}`. This is a manual sanity check — the integration test (Task 7) covers this properly.

**Note on Connect interface:** If the build fails because `BusServer` doesn't satisfy the generated `BusServiceHandler` interface, add `_ chaosv1connect.BusServiceHandler = (*BusServer)(nil)` as a compile-time check in `server.go`, and embed `chaosv1connect.UnimplementedBusServiceHandler` if the generated code requires it.

- [ ] **Step 5: Commit**

```bash
git add bus/internal/server.go bus/main.go
git commit -m "feat(bus): add Connect gRPC server and main binary"
```

### Task 7: Go Integration Test

**Context:** End-to-end test within Go: starts the full server, publishes events via HTTP, subscribes via Connect gRPC, verifies delivery order and replay, sends commands, verifies file output.

**Files:**
- Create: `bus/internal/integration_test.go`

- [ ] **Step 1: Write the integration test**

```go
// bus/internal/integration_test.go
package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	chaosv1 "chaos-bus/gen/chaos/v1"
	"chaos-bus/gen/chaos/v1/chaosv1connect"

	"connectrpc.com/connect"
)

// startTestServer creates a full bus server on a random port and returns
// the base URL and a cleanup function.
func startTestServer(t *testing.T, outputFile string) (string, func()) {
	t.Helper()

	broker := NewBroker()
	queue := NewCommandQueue()

	mux := http.NewServeMux()
	mux.HandleFunc("/publish", NewPublishHandler(broker, queue))
	mux.HandleFunc("/commands", NewCommandsHandler(queue))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	busServer := NewBusServer(broker, queue)
	path, handler := chaosv1connect.NewBusServiceHandler(busServer)
	mux.Handle(path, handler)

	if outputFile != "" {
		cancel := StartFileWriter(broker, outputFile)
		t.Cleanup(cancel)
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	server := &http.Server{Handler: mux}
	go server.Serve(listener)

	port := listener.Addr().(*net.TCPAddr).Port
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)

	cleanup := func() {
		server.Close()
	}

	return baseURL, cleanup
}

func postEvent(baseURL string, payload map[string]any) error {
	body, _ := json.Marshal(payload)
	resp, err := http.Post(baseURL+"/publish", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

func getCommands(baseURL string) ([]map[string]any, error) {
	resp, err := http.Get(baseURL + "/commands")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var cmds []map[string]any
	err = json.Unmarshal(body, &cmds)
	return cmds, err
}

func TestIntegration_PublishAndSubscribe(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	// Publish events
	if err := postEvent(baseURL, map[string]any{
		"event":  "run_started",
		"run_id": "run-test",
	}); err != nil {
		t.Fatal(err)
	}
	if err := postEvent(baseURL, map[string]any{
		"event":  "worker_completed",
		"run_id": "run-test",
		"metric": 2.5,
	}); err != nil {
		t.Fatal(err)
	}

	// Subscribe via Connect gRPC — should replay both events
	client := chaosv1connect.NewBusServiceClient(
		http.DefaultClient,
		baseURL,
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.Subscribe(ctx, connect.NewRequest(&chaosv1.SubscribeRequest{}))
	if err != nil {
		t.Fatal(err)
	}

	var received []string
	for i := 0; i < 2; i++ {
		if !stream.Receive() {
			t.Fatalf("stream ended early at event %d: %v", i, stream.Err())
		}
		received = append(received, stream.Msg().EventType)
	}

	if received[0] != "run_started" || received[1] != "worker_completed" {
		t.Errorf("got %v, want [run_started, worker_completed]", received)
	}
}

func TestIntegration_Commands(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	// Set current run via publish
	postEvent(baseURL, map[string]any{
		"event":  "run_started",
		"run_id": "run-cmd",
	})

	// Send PauseRun via gRPC
	client := chaosv1connect.NewBusServiceClient(
		http.DefaultClient,
		baseURL,
	)
	_, err := client.PauseRun(context.Background(), connect.NewRequest(&chaosv1.PauseRunRequest{}))
	if err != nil {
		t.Fatal(err)
	}

	// Poll commands via HTTP
	cmds, err := getCommands(baseURL)
	if err != nil {
		t.Fatal(err)
	}
	if len(cmds) != 1 {
		t.Fatalf("got %d commands, want 1", len(cmds))
	}
	if cmds[0]["command"] != "pause" {
		t.Errorf("got command %v, want pause", cmds[0]["command"])
	}

	// Second poll should be empty
	cmds2, _ := getCommands(baseURL)
	if len(cmds2) != 0 {
		t.Errorf("got %d commands after drain, want 0", len(cmds2))
	}
}

func TestIntegration_FileWriter(t *testing.T) {
	outputPath := t.TempDir() + "/events.jsonl"
	baseURL, cleanup := startTestServer(t, outputPath)
	defer cleanup()

	postEvent(baseURL, map[string]any{
		"event":         "worker_completed",
		"run_id":        "run-file",
		"experiment_id": "exp-0",
		"metric":        3.14,
	})

	// Wait for file writer to flush
	time.Sleep(200 * time.Millisecond)

	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}

	var record map[string]any
	json.Unmarshal([]byte(lines[0]), &record)

	if record["event"] != "worker_completed" {
		t.Errorf("event: got %v", record["event"])
	}
	if record["run_id"] != "run-file" {
		t.Errorf("run_id: got %v", record["run_id"])
	}
	if record["experiment_id"] != "exp-0" {
		t.Errorf("experiment_id: got %v", record["experiment_id"])
	}
}

func TestIntegration_Healthz(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	resp, err := http.Get(baseURL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("got status %d", resp.StatusCode)
	}

	var body map[string]string
	json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("got status %v", body["status"])
	}
}

func TestIntegration_ExtendBudget(t *testing.T) {
	baseURL, cleanup := startTestServer(t, "")
	defer cleanup()

	postEvent(baseURL, map[string]any{
		"event":  "run_started",
		"run_id": "run-ext",
	})

	client := chaosv1connect.NewBusServiceClient(
		http.DefaultClient,
		baseURL,
	)
	_, err := client.ExtendBudget(context.Background(), connect.NewRequest(
		&chaosv1.ExtendBudgetRequest{
			AddCostUsd:     10.0,
			AddExperiments: 5,
		},
	))
	if err != nil {
		t.Fatal(err)
	}

	cmds, _ := getCommands(baseURL)
	if len(cmds) != 1 {
		t.Fatalf("got %d commands, want 1", len(cmds))
	}
	if cmds[0]["command"] != "extend_budget" {
		t.Errorf("got command %v", cmds[0]["command"])
	}
	if cmds[0]["add_cost_usd"] != 10.0 {
		t.Errorf("add_cost_usd: got %v", cmds[0]["add_cost_usd"])
	}
}
```

- [ ] **Step 2: Run all Go tests**

Run: `cd bus && go test ./internal/ -v`
Expected: All tests PASS (broker: 6, publisher: 7, commands: 6, filewriter: 4, integration: 5 = 28 total)

- [ ] **Step 3: Commit**

```bash
git add bus/internal/integration_test.go
git commit -m "test(bus): add Go integration tests for full bus server"
```

---

## Chunk 4: Python Integration

### Task 8: EventPublisher

**Context:** `EventPublisher` replaces `EventLogger` as the coordinator's event sink. Same `log(Event)` interface — the coordinator doesn't change. Posts events to the bus via HTTP. Falls back to `EventLogger` when bus is unreachable. `poll_commands()` polls the bus for remote commands. `read_events()` always reads from the JSONL file (written by bus or by fallback `EventLogger`).

**Important:** Uses only `urllib.request` from stdlib — no new dependencies.

**Files:**
- Create: `chaosengineer/metrics/publisher.py`
- Test: `tests/test_publisher.py`

**Ref:** `chaosengineer/metrics/logger.py` (EventLogger + Event dataclass)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_publisher.py
"""Tests for EventPublisher."""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

import pytest

from chaosengineer.metrics.logger import Event
from chaosengineer.metrics.publisher import EventPublisher


class FakeBusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records published events and serves commands."""

    events = []
    commands = []

    def do_POST(self):
        if self.path == "/publish":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            FakeBusHandler.events.append(body)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/commands":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            cmds = FakeBusHandler.commands
            FakeBusHandler.commands = []
            self.wfile.write(json.dumps(cmds).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


@pytest.fixture()
def fake_bus():
    """Start a fake bus HTTP server on a random port."""
    FakeBusHandler.events = []
    FakeBusHandler.commands = []
    server = HTTPServer(("127.0.0.1", 0), FakeBusHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestEventPublisher:
    def test_log_posts_to_bus(self, fake_bus, tmp_path):
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        pub.log(Event(event="run_started", data={"run_id": "run-1", "workload": "test"}))

        assert len(FakeBusHandler.events) == 1
        evt = FakeBusHandler.events[0]
        assert evt["event"] == "run_started"
        assert evt["run_id"] == "run-1"
        assert evt["workload"] == "test"
        assert "ts" in evt

    def test_log_includes_run_id_from_run_started(self, fake_bus, tmp_path):
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        pub.log(Event(event="run_started", data={"run_id": "run-abc"}))
        pub.log(Event(event="worker_completed", data={"metric": 2.5}))

        # Second event should include run_id from run_started
        evt = FakeBusHandler.events[1]
        assert evt["run_id"] == "run-abc"

    def test_fallback_when_bus_unavailable(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        pub = EventPublisher(bus_url=None, fallback_path=log_path)
        pub.log(Event(event="test_event", data={"key": "value"}))

        assert log_path.exists()
        records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["event"] == "test_event"

    def test_fallback_on_connection_error(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        # Point to a port that nothing listens on
        pub = EventPublisher(
            bus_url="http://127.0.0.1:1",
            fallback_path=log_path,
        )
        pub.log(Event(event="test_event", data={"key": "value"}))

        assert log_path.exists()

    def test_poll_commands(self, fake_bus, tmp_path):
        FakeBusHandler.commands = [
            {"command": "pause", "run_id": "run-1"},
        ]
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        cmds = pub.poll_commands()

        assert len(cmds) == 1
        assert cmds[0]["command"] == "pause"

    def test_poll_commands_empty_when_no_bus(self, tmp_path):
        pub = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl")
        assert pub.poll_commands() == []

    def test_read_events_from_file(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        log_path.write_text(
            '{"ts":"2026-01-01T00:00:00+00:00","event":"run_started","run_id":"r1"}\n'
            '{"ts":"2026-01-01T00:00:01+00:00","event":"worker_completed","metric":1.0}\n'
        )
        pub = EventPublisher(bus_url=None, fallback_path=log_path)

        all_events = pub.read_events()
        assert len(all_events) == 2

        starts = pub.read_events("run_started")
        assert len(starts) == 1
        assert starts[0]["run_id"] == "r1"

    def test_read_events_empty_when_no_file(self, tmp_path):
        pub = EventPublisher(bus_url=None, fallback_path=tmp_path / "missing.jsonl")
        assert pub.read_events() == []

    def test_raises_when_no_bus_and_no_fallback(self):
        with pytest.raises(RuntimeError):
            EventPublisher(bus_url=None, fallback_path=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_publisher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chaosengineer.metrics.publisher'`

- [ ] **Step 3: Implement EventPublisher**

```python
# chaosengineer/metrics/publisher.py
"""Event publisher that posts to the message bus with fallback to EventLogger."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chaosengineer.metrics.logger import Event, EventLogger


class EventPublisher:
    """Publishes events to the message bus, falling back to direct file writes.

    Same log(Event) interface as EventLogger — the coordinator doesn't change.
    """

    def __init__(self, bus_url: str | None, fallback_path: Path | None = None):
        self._bus_url = bus_url
        self._fallback_path = fallback_path
        self._fallback: EventLogger | None = None
        self._bus_available = False
        self._run_id: str | None = None

        if bus_url:
            try:
                req = urllib.request.Request(f"{bus_url}/healthz")
                urllib.request.urlopen(req, timeout=2)
                self._bus_available = True
            except Exception:
                pass

        if not self._bus_available:
            if fallback_path:
                self._fallback = EventLogger(fallback_path)
            else:
                raise RuntimeError(
                    "Bus unreachable and no fallback path provided"
                )

    def log(self, event: Event) -> None:
        """Publish an event to the bus, or fall back to direct file write."""
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {"ts": ts, "event": event.event, **event.data}

        # Track run_id from run_started events
        if event.event == "run_started" and "run_id" in event.data:
            self._run_id = event.data["run_id"]

        # Include run_id on all events if known
        if "run_id" not in payload and self._run_id:
            payload["run_id"] = self._run_id

        if not self._bus_available:
            if self._fallback:
                self._fallback.log(event)
            return

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._bus_url}/publish",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            # Fall back to direct file write on connection error
            if self._fallback:
                self._fallback.log(event)
            else:
                import sys
                print(
                    f"Warning: failed to publish event {event.event}",
                    file=sys.stderr,
                )

    def poll_commands(self) -> list[dict]:
        """Poll the bus for pending commands. Returns empty list on error."""
        if not self._bus_available:
            return []
        try:
            req = urllib.request.Request(f"{self._bus_url}/commands")
            resp = urllib.request.urlopen(req, timeout=2)
            return json.loads(resp.read())
        except Exception:
            return []

    def read_events(self, event_type: str | None = None) -> list[dict]:
        """Read events from the JSONL file (written by bus or fallback).

        Always reads from the file — works whether the bus or EventLogger wrote it.
        """
        if self._fallback_path is None or not self._fallback_path.exists():
            return []
        events = []
        with open(self._fallback_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if event_type is None or record.get("event") == event_type:
                    events.append(record)
        return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_publisher.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/ -v --ignore=tests/e2e`
Expected: All existing tests PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/metrics/publisher.py tests/test_publisher.py
git commit -m "feat(bus): add EventPublisher with bus posting and fallback"
```

### Task 9: Coordinator Command Polling

**Context:** The coordinator needs to poll the bus for remote commands (pause, extend_budget) at its existing pause checkpoints. Add a `_poll_bus_commands()` helper that's called at the top of the main loop and after each iteration. Uses duck-typing (`hasattr`) so it works with both `EventPublisher` and `EventLogger`.

**Files:**
- Modify: `chaosengineer/core/coordinator.py`
- Test: `tests/test_bus_commands.py`

**Ref:** Pause checkpoints are at `coordinator.py:131` (top of while loop) and `coordinator.py:224` (after iteration). The `BudgetConfig` reconstruction pattern is at `cli.py:326-347`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bus_commands.py
"""Tests for coordinator command polling from the message bus."""
from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
)
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import Event, EventLogger


def make_coordinator(logger, budget_config=None):
    """Create a minimal coordinator for testing command polling."""
    from chaosengineer.core.pause import PauseController

    spec = MagicMock()
    spec.name = "test"
    spec.dimensions = []
    spec.budget = budget_config or BudgetConfig(max_experiments=100)
    spec.primary_metric = "loss"
    spec.metric_direction = "lower"
    spec.execution_command = "echo test"
    spec.spec_hash.return_value = "abc123"

    budget = BudgetTracker(spec.budget)
    pause_controller = PauseController()

    coordinator = Coordinator(
        spec=spec,
        decision_maker=MagicMock(),
        executor=MagicMock(),
        logger=logger,
        budget=budget,
        initial_baseline=Baseline("HEAD", 1.0, "loss"),
        pause_controller=pause_controller,
    )
    return coordinator, pause_controller


class TestPollBusCommands:
    def test_pause_command_sets_flag(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "pause", "run_id": "run-1"}
        ]

        coord, pause_ctrl = make_coordinator(logger)
        coord._poll_bus_commands()

        assert pause_ctrl.pause_requested is True

    def test_extend_budget_cost(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 10.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_api_cost=5.0, max_experiments=10)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 15.0
        assert coord.budget.config.max_experiments == 10  # unchanged

    def test_extend_budget_experiments(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_experiments": 5}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_experiments=10)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_experiments == 15

    def test_extend_budget_time(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_time_seconds": 300.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_wall_time_seconds=600.0)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_wall_time_seconds == 900.0

    def test_extend_budget_from_none(self):
        """Extending a None budget field should start from 0."""
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 10.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_api_cost=None)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 10.0

    def test_no_poll_when_logger_lacks_method(self):
        """EventLogger doesn't have poll_commands — should be a no-op."""
        logger = EventLogger("/dev/null")
        coord, pause_ctrl = make_coordinator(logger)
        coord._poll_bus_commands()  # Should not raise

        assert pause_ctrl.pause_requested is False

    def test_multiple_commands_in_one_poll(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 5.0},
            {"command": "pause", "run_id": "run-1"},
        ]

        coord, pause_ctrl = make_coordinator(
            logger, BudgetConfig(max_api_cost=10.0)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 15.0
        assert pause_ctrl.pause_requested is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bus_commands.py -v`
Expected: FAIL — `AttributeError: 'Coordinator' object has no attribute '_poll_bus_commands'`

- [ ] **Step 3: Add _poll_bus_commands to Coordinator**

First, add `BudgetConfig` to the existing import from `chaosengineer.core.models` at the top of `coordinator.py` (around line 9):

```python
from chaosengineer.core.models import (..., BudgetConfig)
```

Then add this method to the `Coordinator` class, after the `_log` method (around line 71):

```python
def _poll_bus_commands(self) -> None:
    """Poll message bus for remote commands (pause, extend_budget)."""
    if not hasattr(self.logger, "poll_commands"):
        return
    for cmd in self.logger.poll_commands():
        if cmd.get("command") == "pause" and self._pause_controller:
            self._pause_controller.pause_requested = True
        elif cmd.get("command") == "extend_budget":
            bc = self.budget.config
            if cmd.get("add_cost_usd"):
                bc = BudgetConfig(
                    max_api_cost=(bc.max_api_cost or 0) + cmd["add_cost_usd"],
                    max_experiments=bc.max_experiments,
                    max_wall_time_seconds=bc.max_wall_time_seconds,
                    max_plateau_iterations=bc.max_plateau_iterations,
                )
            if cmd.get("add_experiments"):
                bc = BudgetConfig(
                    max_api_cost=bc.max_api_cost,
                    max_experiments=(bc.max_experiments or 0) + cmd["add_experiments"],
                    max_wall_time_seconds=bc.max_wall_time_seconds,
                    max_plateau_iterations=bc.max_plateau_iterations,
                )
            if cmd.get("add_time_seconds"):
                bc = BudgetConfig(
                    max_api_cost=bc.max_api_cost,
                    max_experiments=bc.max_experiments,
                    max_wall_time_seconds=(bc.max_wall_time_seconds or 0) + cmd["add_time_seconds"],
                    max_plateau_iterations=bc.max_plateau_iterations,
                )
            self.budget.config = bc
```

Then add two calls to `_poll_bus_commands()` in `_run_loop()`:

**Call 1:** At the top of the while loop, before the pre-iteration pause check (before line 133):
```python
while not self.budget.is_exhausted():
    # Poll bus for remote commands
    self._poll_bus_commands()

    # Pause check: before starting new iteration
    if self._pause_controller and ...
```

**Call 2:** After iteration completion, before the post-iteration pause check (before line 229):
```python
                # Status display: iteration done
                if self._status_display:
                    self._status_display.on_iteration_done(...)

                # Poll bus for remote commands
                self._poll_bus_commands()

                # Pause check: auto-pause after kill
                if self._pause_controller and self._pause_controller.kill_issued:
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_bus_commands.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/ -v --ignore=tests/e2e`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_bus_commands.py
git commit -m "feat(bus): add coordinator command polling at pause checkpoints"
```

### Task 10: CLI Bus Spawning

**Context:** The CLI spawns the bus binary as a subprocess before creating the coordinator. If the binary isn't found, falls back to `EventPublisher` with `bus_url=None` (which uses `EventLogger` internally). Both `_execute_run()` and `_execute_resume()` need this.

**Key behaviors:**
- Binary discovery: `CHAOS_BUS_BIN` env var > `bus/chaos-bus` relative to repo > `chaos-bus` on PATH
- Read port from bus stdout (JSON: `{"port": N}`)
- Poll `/healthz` until ready
- SIGTERM the bus after `coordinator.run()` returns

**Files:**
- Modify: `chaosengineer/cli.py`

- [ ] **Step 1: Add helper functions to cli.py**

Add these before `_execute_run()` (around line 173):

```python
def _find_bus_binary() -> Path | None:
    """Locate the chaos-bus binary: env var > repo path > system PATH."""
    import shutil
    env_path = os.environ.get("CHAOS_BUS_BIN")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    # Development: relative to repo root (cli.py is in chaosengineer/)
    repo_root = Path(__file__).parent.parent
    dev_path = repo_root / "bus" / "chaos-bus"
    if dev_path.is_file():
        return dev_path
    found = shutil.which("chaos-bus")
    if found:
        return Path(found)
    return None


def _start_bus(output_file: Path) -> tuple[subprocess.Popen | None, str | None]:
    """Start the bus binary and return (process, bus_url) or (None, None)."""
    binary = _find_bus_binary()
    if binary is None:
        return None, None

    proc = subprocess.Popen(
        [str(binary), "--port", "0", "--output-file", str(output_file),
         "--shutdown-delay", "5s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        port_line = proc.stdout.readline()
        if not port_line:
            proc.kill()
            return None, None
        port_data = json.loads(port_line)
        bus_url = f"http://127.0.0.1:{port_data['port']}"

        # Poll healthz until ready (up to 5 seconds)
        import urllib.request
        import time
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{bus_url}/healthz", timeout=1)
                return proc, bus_url
            except Exception:
                time.sleep(0.1)

        proc.kill()
        return None, None
    except Exception:
        proc.kill()
        return None, None
```

- [ ] **Step 2: Modify _execute_run() to use the bus**

In `_execute_run()`, replace the logger creation (around line 243):

**Before:**
```python
    logger = EventLogger(args.output_dir / "events.jsonl")
```

**After:**
```python
    from chaosengineer.metrics.publisher import EventPublisher
    log_path = args.output_dir / "events.jsonl"
    bus_proc, bus_url = _start_bus(log_path)
    logger = EventPublisher(bus_url=bus_url, fallback_path=log_path)
```

Then wrap the coordinator.run() in a try/finally that terminates the bus:

**Before:**
```python
    pause_controller.install()
    try:
        coordinator.run()
    finally:
        pause_controller.uninstall()
```

**After:**
```python
    pause_controller.install()
    try:
        coordinator.run()
    finally:
        pause_controller.uninstall()
        if bus_proc:
            bus_proc.terminate()
```

Also add `import os`, `import subprocess`, and `import json` at the top of the file if not already present. (`os` is needed by `_find_bus_binary` for `os.environ`.)

- [ ] **Step 3: Modify _execute_resume() the same way**

In `_execute_resume()`, replace the logger creation (around line 416):

**Before:**
```python
    logger = EventLogger(events_path)
```

**After:**
```python
    from chaosengineer.metrics.publisher import EventPublisher
    bus_proc, bus_url = _start_bus(events_path)
    logger = EventPublisher(bus_url=bus_url, fallback_path=events_path)
```

And wrap the resume call:

**Before:**
```python
    pause_controller.install()
    try:
        coordinator.resume_from_snapshot(...)
    finally:
        pause_controller.uninstall()
```

**After:**
```python
    pause_controller.install()
    try:
        coordinator.resume_from_snapshot(...)
    finally:
        pause_controller.uninstall()
        if bus_proc:
            bus_proc.terminate()
```

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `pytest tests/ -v --ignore=tests/e2e`
Expected: All tests PASS (bus binary not found → graceful fallback → same behavior as before)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli.py
git commit -m "feat(bus): wire bus subprocess into CLI run and resume commands"
```

### Task 11: Python-Go E2E Integration Test

**Context:** A single end-to-end test that exercises the full pipeline: build the Go binary, start the bus, run a scripted coordinator scenario, subscribe via the bus, and verify events arrive in the JSONL file. Skipped when the bus binary isn't built.

**Files:**
- Create: `tests/e2e/test_bus_integration.py`

- [ ] **Step 1: Write the E2E test**

```python
# tests/e2e/test_bus_integration.py
"""End-to-end integration test for the message bus.

Requires the bus binary to be built: cd bus && go build -o chaos-bus .
Skipped automatically if the binary is not found.
"""
import json
import subprocess
import time
from pathlib import Path

import pytest

from chaosengineer.metrics.logger import Event
from chaosengineer.metrics.publisher import EventPublisher


def find_bus_binary() -> Path | None:
    repo_root = Path(__file__).parent.parent.parent
    binary = repo_root / "bus" / "chaos-bus"
    if binary.is_file():
        return binary
    return None


BUS_BINARY = find_bus_binary()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(BUS_BINARY is None, reason="bus binary not built"),
]


@pytest.fixture()
def bus_server(tmp_path):
    """Start the bus binary and yield (bus_url, events_path)."""
    events_path = tmp_path / "events.jsonl"
    proc = subprocess.Popen(
        [str(BUS_BINARY), "--port", "0", "--output-file", str(events_path),
         "--shutdown-delay", "1s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    port_line = proc.stdout.readline()
    port_data = json.loads(port_line)
    bus_url = f"http://127.0.0.1:{port_data['port']}"

    # Wait for healthy
    import urllib.request
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{bus_url}/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Bus did not become healthy")

    yield bus_url, events_path

    proc.terminate()
    proc.wait(timeout=10)


class TestBusIntegration:
    def test_publish_and_read_events(self, bus_server, tmp_path):
        bus_url, events_path = bus_server

        pub = EventPublisher(bus_url=bus_url, fallback_path=events_path)

        # Publish a sequence of events
        pub.log(Event(event="run_started", data={
            "run_id": "run-e2e",
            "workload": "test",
            "budget": {},
            "baseline": {"commit": "HEAD", "metric_value": 1.0},
            "mode": "sequential",
            "metric_direction": "lower",
            "workload_spec_hash": "abc",
        }))
        pub.log(Event(event="worker_completed", data={
            "experiment_id": "exp-0-0",
            "dimension": "lr",
            "metric": 2.5,
            "cost_usd": 0.12,
        }))
        pub.log(Event(event="run_completed", data={
            "reason": "all_dimensions_explored",
        }))

        # Wait for file writer to flush
        time.sleep(0.5)

        # Read events back from the JSONL file (written by bus)
        events = pub.read_events()
        assert len(events) == 3
        assert events[0]["event"] == "run_started"
        assert events[0]["run_id"] == "run-e2e"
        assert events[1]["event"] == "worker_completed"
        assert events[1]["metric"] == 2.5
        assert events[2]["event"] == "run_completed"

    def test_poll_commands_empty(self, bus_server):
        """Verify poll_commands returns empty when no commands are queued."""
        bus_url, events_path = bus_server

        pub = EventPublisher(bus_url=bus_url, fallback_path=events_path)
        pub.log(Event(event="run_started", data={"run_id": "run-cmd"}))

        # No commands have been queued, poll should return empty
        assert pub.poll_commands() == []

        # Note: The gRPC → command queue → poll path is tested in the
        # Go integration tests (Task 7). Python cannot easily call
        # Connect gRPC without a dedicated client library.
```

- [ ] **Step 2: Build the bus binary**

Run: `cd bus && go build -o chaos-bus . && cd ..`

- [ ] **Step 3: Run the E2E test**

Run: `pytest tests/e2e/test_bus_integration.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/e2e && cd bus && go test ./internal/ -v && cd ..`
Expected: All Python tests PASS, all Go tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_bus_integration.py
git commit -m "test(bus): add Python-Go E2E integration test"
```
