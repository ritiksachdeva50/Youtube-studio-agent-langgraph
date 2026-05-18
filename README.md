# yt-studio-simple

Minimal LangGraph multi-agent YouTube video generator. No MCP, no extras.

## Setup
Create an env file and put your openai api key.
```powershell
conda activate yt-studio        # or any Python 3.11 env
pip install -r requirements.txt
```

## Run
```powershell
python main.py "How LangGraph beats prompt chains"
```

Output goes to `output/video/final.mp4` and `output/images/`.

## Files
- `main.py`   — CLI entry
- `agents.py` — LangGraph pipeline (researcher → scriptwriter → seo → producer)
- `media.py`  — OpenAI image / TTS + MoviePy assembly

## Demo
- A demo is provided in the output folder on prompt "Why One Piece is worth watching?"
