# Create and activate venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip, install uv
py -m pip install --upgrade pip
py -m pip install uv

# Install packages (runtime + tools)
uv pip install openai tiktoken llama-index datasets ruff pytest