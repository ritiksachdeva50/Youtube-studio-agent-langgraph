"""LangGraph multi-agent pipeline: research -> script -> seo -> produce."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import media


# ---------- Schemas ----------
class Research(BaseModel):
    summary: str
    key_points: list[str] = Field(min_length=3)


class Scene(BaseModel):
    narration: str
    visual_prompt: str


class Script(BaseModel):
    title: str
    scenes: list[Scene] = Field(min_length=2, max_length=5)


class SEO(BaseModel):
    title: str = Field(max_length=100)
    description: str
    tags: list[str] = Field(max_length=15)
    thumbnail_prompt: str


class State(TypedDict, total=False):
    topic: str
    research: Research
    script: Script
    seo: SEO
    video_path: str
    thumbnail_path: str
    messages: Annotated[list, add_messages]


# ---------- LLM ----------
def llm(temperature: float = 0.4):
    return init_chat_model("gpt-4o-mini", model_provider="openai", temperature=temperature)


# ---------- Nodes ----------
def researcher(state: State) -> dict:
    out = llm(0.2).with_structured_output(Research).invoke([
        SystemMessage(content="You are a research analyst. Produce a tight brief."),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    return {"research": out}


def scriptwriter(state: State) -> dict:
    r = state["research"]
    out = llm(0.6).with_structured_output(Script).invoke([
        SystemMessage(
            content="Write a short YouTube video script. 2-4 scenes. "
                    "Each scene has 1-2 sentences of narration and a vivid visual_prompt."
        ),
        HumanMessage(content=f"Topic: {state['topic']}\nBrief: {r.summary}\nPoints: {r.key_points}"),
    ])
    return {"script": out}


def seo(state: State) -> dict:
    s = state["script"]
    out = llm(0.5).with_structured_output(SEO).invoke([
        SystemMessage(content="Create YouTube SEO metadata and a 16:9 thumbnail prompt."),
        HumanMessage(content=f"Title: {s.title}\nScenes: {[sc.narration for sc in s.scenes]}"),
    ])
    return {"seo": out}


def producer(state: State) -> dict:
    script = state["script"]
    seo_data = state["seo"]
    images = [media.generate_image(sc.visual_prompt) for sc in script.scenes]
    audios = [media.synthesize_speech(sc.narration) for sc in script.scenes]
    thumb = media.generate_image(seo_data.thumbnail_prompt, size="1536x1024")
    video = media.assemble_video(images, audios, name="final.mp4")
    return {"video_path": video, "thumbnail_path": thumb}


# ---------- Graph ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("researcher", researcher)
    g.add_node("scriptwriter", scriptwriter)
    g.add_node("seo", seo)
    g.add_node("producer", producer)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "scriptwriter")
    g.add_edge("scriptwriter", "seo")
    g.add_edge("seo", "producer")
    g.add_edge("producer", END)
    return g.compile()
