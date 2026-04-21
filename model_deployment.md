# Model Deployment Details

## Model Information
- Model name: mistral:7b-instruct
- Model family: Mistral
- Parameter class: 7B
- Quantization: 4-bit (default Ollama)

## Serving Stack
- Tool: Ollama v0.6.1
- Command used: ollama pull mistral:7b-instruct
- Server: ollama serve (runs as background process on Mac)
- API endpoint: http://localhost:11434

## Hardware / Runtime Environment
- Machine: MacBook (Apple Silicon / Intel)
- RAM: System RAM (no dedicated GPU)
- OS: macOS

## Latency Observations
- Typical retrieval latency: 300-2000ms
- Typical generation latency: 10-30s per query on CPU
- End-to-end latency: 11-32s per query

## Startup Commands
ollama serve
ollama pull mistral:7b-instruct
ollama run mistral:7b-instruct
