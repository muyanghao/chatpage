
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"


vllm serve "openai/gpt-oss-20b" --gpu-memory-utilization 0.2
