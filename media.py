"""Image generation, text-to-speech, and video assembly. Direct OpenAI + MoviePy."""
from __future__ import annotations

import base64
import uuid
from pathlib import Path

from openai import OpenAI

OUT = Path("output")
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "audio").mkdir(parents=True, exist_ok=True)
(OUT / "video").mkdir(parents=True, exist_ok=True)


def _client() -> OpenAI:
    return OpenAI(timeout=120.0)


def generate_image(prompt: str, size: str = "1024x1024") -> str:
    resp = _client().images.generate(model="gpt-image-1", prompt=prompt, size=size)
    path = OUT / "images" / f"img_{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(base64.b64decode(resp.data[0].b64_json))
    return str(path)


def synthesize_speech(text: str, voice: str = "alloy") -> str:
    resp = _client().audio.speech.create(
        model="gpt-4o-mini-tts", voice=voice, input=text, response_format="mp3"
    )
    path = OUT / "audio" / f"vo_{uuid.uuid4().hex[:8]}.mp3"
    path.write_bytes(resp.read())
    return str(path)


def assemble_video(image_paths: list[str], audio_paths: list[str], name: str = "final.mp4") -> str:
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

    clips, audios = [], []
    for img, aud in zip(image_paths, audio_paths):
        a = AudioFileClip(aud)
        audios.append(a)
        clips.append(
            ImageClip(img).resized(new_size=(1280, 720)).with_duration(a.duration).with_audio(a)
        )
    final = concatenate_videoclips(clips, method="chain")
    out = OUT / "video" / name
    final.write_videofile(
        str(out), fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4
    )
    final.close()
    for c in clips:
        c.close()
    for a in audios:
        a.close()
    return str(out)
