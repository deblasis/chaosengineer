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
