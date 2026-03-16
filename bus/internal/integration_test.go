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
	mux.HandleFunc("/events", NewEventsHandler(broker))
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

	// Read first line -- should be the replayed event
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
