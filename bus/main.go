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
