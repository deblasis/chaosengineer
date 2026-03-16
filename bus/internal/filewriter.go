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
