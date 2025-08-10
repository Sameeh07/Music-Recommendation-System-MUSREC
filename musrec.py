import os
import json
import base64
import html
import re
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
    "6) Only new songs (post 2000 era), no old songs. \n"
    "7) Make sure it's a hindi/bollywood song, not any other video. \n"
)

# ---------------------------
# Helpers
# ---------------------------

def encode_image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@dataclass
class Song:
    title: str
    artist: str
    film_or_album: str
    year: int
    vibe_tags: List[str]
    yt_search_query: str


# ---------- YouTube matching that respects title + artist + film ----------

PREFERRED_CHANNEL_KEYWORDS = [
    "T-Series", "Zee Music", "Sony Music", "YRF", "Saregama",
    "Tips Official", "Eros", "Dharma", "SVF", "Shemaroo", "Junglee Music",
]

PENALTY_TERMS = [
    "cover", "slowed", "sped", "sped up", "8d", "nightcore", "remix",
    "dance", "status", "shorts", "live", "edit", "reverb", "lofi",
]

BOOST_TERMS = [
    "official", "video", "audio", "lyric"
]

def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _tokens_for_song(song: Song) -> List[str]:
    parts = []
    if song.title: parts += song.title.split()
    if song.artist: parts += song.artist.split()
    if song.film_or_album: parts += song.film_or_album.split()
    toks = [_norm_text(t) for t in parts if t]
    seen, out = set(), []
    for t in toks:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _score_candidate(song: Song, video_title: str, channel_title: str) -> int:
    vt = _norm_text(video_title)
    ct = (channel_title or "").lower()

    score = 0

    # Strong boost if full song title appears as a phrase
    if _norm_text(song.title) and _norm_text(song.title) in vt:
        score += 5

    # Token coverage from title/artist/film
    tokens = _tokens_for_song(song)
    score += sum(1 for t in tokens if t and t in vt)

    # Prefer official label channels
    if any(k.lower() in ct for k in (kw.lower() for kw in PREFERRED_CHANNEL_KEYWORDS)):
        score += 4

    # Boost likely official/audio/lyric content
    score += sum(1 for b in BOOST_TERMS if b in vt)

    # Penalize derivatives/covers/edits
    score -= sum(2 for p in PENALTY_TERMS if p in vt)

    return score

def fetch_best_youtube_video_id_for_song(song: Song) -> Optional[str]:
    """
    Use YouTube Data API to find the best embeddable video matching the specific song.
    Searches with title+artist+film, scores candidates, and returns a videoId.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return None

    base_q = " ".join(x for x in [song.title, song.artist, song.film_or_album] if x).strip()
    if not base_q:
        base_q = song.yt_search_query or ""

    queries = [
        base_q,
        f"{base_q} official video",
        f"{base_q} audio",
        f"{base_q} lyric video",
    ]

    try:
        for q in queries:
            # Need title+channel for scoring => part=snippet
            search_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "key": api_key,
                "q": q,
                "part": "snippet",
                "type": "video",
                "maxResults": 12,
                "videoEmbeddable": "true",
                "regionCode": "IN",
                "relevanceLanguage": "hi",
                "order": "relevance",
                "safeSearch": "none",
            }
            sresp = requests.get(search_url, params=params, timeout=10)
            sresp.raise_for_status()
            sdata = sresp.json()
            candidates = [
                (
                    item["id"]["videoId"],
                    item["snippet"]["title"],
                    item["snippet"].get("channelTitle", "")
                )
                for item in sdata.get("items", [])
                if item.get("id", {}).get("videoId")
            ]
            if not candidates:
                continue

            # Verify embeddable (extra safety)
            videos_url = "https://www.googleapis.com/youtube/v3/videos"
            vparams = {
                "key": api_key,
                "id": ",".join([vid for vid, _, _ in candidates]),
                "part": "status",
            }
            vresp = requests.get(videos_url, params=vparams, timeout=10)
            vresp.raise_for_status()
            vdata = vresp.json()
            emb_ok = {it["id"]: it["status"].get("embeddable", True) for it in vdata.get("items", [])}

            # Score and pick best
            best_vid, best_score = None, -10**9
            for vid, title, ch in candidates:
                if not emb_ok.get(vid, True):
                    continue
                sc = _score_candidate(song, title, ch)
                if sc > best_score:
                    best_vid, best_score = vid, sc

            if best_vid:
                return best_vid

    except Exception:
        pass

    return None

def build_youtube_iframe_for_song(song: Song, height: int = 360) -> str:
    """
    Prefer a specific videoId chosen using structured song fields; fall back to a search playlist.
    """
    vid = fetch_best_youtube_video_id_for_song(song)
    if vid:
        src = f"https://www.youtube.com/embed/{vid}?rel=0"
        link = f"https://www.youtube.com/watch?v={vid}"
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
    # Last resort: search playlist for structured query
    q = quote_plus(" ".join(x for x in [song.title, song.artist, song.film_or_album, "Bollywood"] if x))
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
        build_youtube_iframe_for_song(s),
    ]

    return "\n".join(blocks)

with gr.Blocks(title="Music Recommendation System") as demo:
    gr.Markdown("""
    # 🎬 MusRec
    Not sure what song fits your Insta story? Upload a pic to MUSREC.
    It reads the mood and recommends a Bollywood track you can play right away.
    """)

    with gr.Row():
        img = gr.Image(label="Upload a photo", type="pil")
        notes = gr.Textbox(
            label="Prompt (e.g., 'monsoon romantic', 'party vibe', '90s nostalgia')",
            placeholder="Add any hints you want the model to consider (optional)"
        )

    with gr.Row():
        go = gr.Button("Recommend 🎵", variant="primary")
        clear_btn = gr.Button("Reset")

    html_out = gr.HTML(label="Player & Reasoning")

    go.click(ui_infer, inputs=[img, notes], outputs=[html_out])
    clear_btn.click(lambda: "<p>Upload an image to begin.</p>", inputs=[], outputs=[html_out])

if __name__ == "__main__":
    os.environ["GRADIO_DISABLE_BROTLI"] = "1"  # avoid rare Windows stack bug locally
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860"))
    )
