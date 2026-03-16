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
