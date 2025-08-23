import os
import json
import base64
import html
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------
# Config
# ---------------------------
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SONG_COUNT = 1  # always 1

SYSTEM_PROMPT = (
    "You are a seasoned Bollywood music curator.\n"
    "Task: Given a user-provided IMAGE (and optional user notes), infer the mood, setting, and context, "
    "then recommend ONLY one Hindi/Bollywood song. Favor popular, recognizable cinema soundtracks. \n\n"
    "Rules:\n"
    "1) Always output STRICT JSON with this schema: {\n"
    "   'mood': str,\n"
    "   'keywords': [str,...],\n"
    "   'reasoning': str,\n"
    "   'songs': [ { 'title': str, 'artist': str, 'film_or_album': str, 'year': int,\n"
    "               'vibe_tags': [str,...], 'yt_search_query': str } ]\n"
    "}\n"
    "2) Return exactly ONE song in 'songs'.\n"
    "3) For the song, craft a concise 'yt_search_query' that will likely yield the official "
    "   or highest-quality audio/video on YouTube (include film or album when known).\n"
    "4) Never include lyrics; avoid copyrighted text.\n"
    "5) Keep 'reasoning' to 1–2 sentences about how the image (and optional notes) informed the selection.\n"
)

# ---------------------------
# Helpers
# ---------------------------

def encode_image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fetch_best_youtube_video_id(query: str) -> Optional[str]:
    """
    Use YouTube Data API to find a likely embeddable, region-available video for the query.
    Tries multiple candidate queries; returns a videoId or None.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return None  # No key -> we'll fallback to search playlist embed

    candidate_queries = [
        query,
        f"{query} official video",
        f"{query} audio",
        f"{query} lyric video",
        f"{query} jukebox",
    ]

    for q in candidate_queries:
        try:
            search_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "key": api_key,
                "q": q,
                "part": "id",
                "type": "video",
                "maxResults": 10,
                "videoEmbeddable": "true",
                "regionCode": "IN",
                "relevanceLanguage": "hi",
                "order": "relevance",
                "safeSearch": "none",
            }
            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            candidates = [
                item["id"]["videoId"]
                for item in data.get("items", [])
                if item.get("id", {}).get("videoId")
            ]
            if not candidates:
                continue

            videos_url = "https://www.googleapis.com/youtube/v3/videos"
            vparams = {
                "key": api_key,
                "id": ",".join(candidates),
                "part": "status",
            }
            vresp = requests.get(videos_url, params=vparams, timeout=10)
            vresp.raise_for_status()
            vdata = vresp.json()

            for item in vdata.get("items", []):
                vid = item.get("id")
                status = item.get("status", {})
                if status.get("embeddable", True):
                    return vid

            return candidates[0]

        except Exception:
            continue

    return None


def build_youtube_iframe(search_query: str, height: int = 360) -> str:
    """
    Prefer a specific videoId via YouTube Data API;
    fallback to search playlist embed and always show a direct link.
    """
    video_id = fetch_best_youtube_video_id(search_query)
    if video_id:
        src = f"https://www.youtube.com/embed/{video_id}?rel=0"
        link = f"https://www.youtube.com/watch?v={video_id}"
        return f'''
        <div style="margin: 8px 0;">
          <iframe width="100%" height="{height}" src="{src}" title="YouTube player"
                  frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowfullscreen></iframe>
          <div style="font-size: 0.9rem; margin-top: 4px;">
            Open on YouTube: <a href="{link}" target="_blank" rel="noopener noreferrer">{link}</a>
          </div>
        </div>
        '''
    q = quote_plus(search_query)
    src = f"https://www.youtube.com/embed?listType=search&list={q}&autoplay=0&rel=0"
    return f'''
    <div style="margin: 8px 0;">
      <iframe width="100%" height="{height}" src="{src}" title="YouTube player"
              frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowfullscreen></iframe>
      <div style="font-size: 0.9rem; margin-top: 4px;">
        Can't see the player? <a href="https://www.youtube.com/results?search_query={q}" target="_blank" rel="noopener noreferrer">Open on YouTube</a>
      </div>
    </div>
    '''


@dataclass
class Song:
    title: str
    artist: str
    film_or_album: str
    year: int
    vibe_tags: List[str]
    yt_search_query: str


# ---------------------------
# Core LLM Call
# ---------------------------

def groq_analyze_and_recommend(image_path: str, user_notes: str = "") -> Tuple[str, str, str, Song]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in environment or .env file.")

    client = Groq(api_key=api_key)

    base_prompt = (
        "Analyze the attached image and recommend strictly one Hindi/Bollywood song. "
        f"Return exactly {SONG_COUNT} song."
    )
    if user_notes and user_notes.strip():
        base_prompt += f"\nUser notes to consider: {user_notes.strip()}"

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.save(image_path, format="JPEG", quality=92)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
        top_p=1,
        max_completion_tokens=900,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)

    mood = data.get("mood", "")
    keywords = data.get("keywords", []) or []
    reasoning = data.get("reasoning", "")

    first = (data.get("songs") or [{}])[0]
    song = Song(
        title=first.get("title", ""),
        artist=first.get("artist", ""),
        film_or_album=first.get("film_or_album", ""),
        year=int(first.get("year", 0) or 0),
        vibe_tags=first.get("vibe_tags", []) or [],
        yt_search_query=first.get("yt_search_query", ""),
    )

    return mood, ", ".join(keywords), reasoning, song


# ---------------------------
# Gradio UI Logic
# ---------------------------

def ui_infer(image: Image.Image, user_notes: str) -> str:
    if image is None:
        return "<p>Please upload an image.</p>"

    tmp_path = "_upload.jpg"
    image.convert("RGB").save(tmp_path, format="JPEG", quality=92)

    try:
        mood, keywords_csv, reasoning, s = groq_analyze_and_recommend(tmp_path, user_notes=user_notes)
    except Exception as e:
        return f"<pre>Error: {html.escape(str(e))}</pre>"

    subtitle = f"{s.title} — {s.artist}"
    if s.film_or_album:
        subtitle += f" ({s.film_or_album}, {s.year or ''})"
    tags = ", ".join(s.vibe_tags)

    q = (s.yt_search_query or f"{s.title} {s.artist} {s.film_or_album} Bollywood audio").strip()

    blocks = [
        f"""
        <div style='line-height:1.5'>
          <h3 style='margin:0 0 8px;'>Mood detected: {html.escape(mood)}</h3>
          <p style='margin:0 0 8px;'><b>Keywords:</b> {html.escape(keywords_csv)}</p>
          <p style='margin:0 0 12px;'>{html.escape(reasoning)}</p>
        </div>
        """,
        f"<h4 style='margin:12px 0 4px;'>{html.escape(subtitle)}</h4>"
        f"<div style='color:#555;font-size:0.95rem;margin:0 0 6px;'>Vibe: {html.escape(tags)}</div>",
        build_youtube_iframe(q),
    ]

    return "\n".join(blocks)


with gr.Blocks(title="Music Recommendation System") as demo:
    gr.Markdown("""
    # 🎬 MusRec
    Upload an image. We'll analyze the vibe and recommend Hindi/Bollywood song — with a player so you can listen right away (via YouTube).
    """)

    with gr.Row():
        img = gr.Image(label="Upload a photo", type="pil")
        notes = gr.Textbox(
            label="Prompt: e.g., 'monsoon romantic', 'party vibe', '90s nostalgia')",
            placeholder="Add any hints you want the model to consider (optional)"
        )

    with gr.Row():
        go = gr.Button("Recommend 🎵", variant="primary")
        clear_btn = gr.Button("Reset")

    html_out = gr.HTML(label="Player & Reasoning")

    go.click(ui_infer, inputs=[img, notes], outputs=[html_out])
    clear_btn.click(lambda: "<p>Upload an image to begin.</p>", inputs=[], outputs=[html_out])



#  for render 
if __name__ == "__main__":
    os.environ["GRADIO_DISABLE_BROTLI"] = "1"  
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860"))
    )

# if __name__ == "__main__":
#     os.environ["GRADIO_DISABLE_BROTLI"] = "1"  # for local testing  
#     demo.launch()
 

