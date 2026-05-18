"""Entry point. Usage: python main.py "your topic here" """
from __future__ import annotations

import sys

from dotenv import load_dotenv

from agents import build_graph

load_dotenv()


def main() -> None:
    topic = " ".join(sys.argv[1:]) or "Why LangGraph beats prompt chains"
    print(f"Topic: {topic}\n")
    result = build_graph().invoke({"topic": topic, "messages": []})
    print("\n--- DONE ---")
    print("Title    :", result["seo"].title)
    print("Scenes   :", len(result["script"].scenes))
    print("Video    :", result["video_path"])
    print("Thumbnail:", result["thumbnail_path"])


if __name__ == "__main__":
    main()
