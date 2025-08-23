Upload a picture, and the app analyzes its vibe (mood, setting, context) using a multimodal LLaMA model and recommends Hindi/Bollywood song. It also embeds a **YouTube player** so you can listen immediately. Clean, simple **Gradio** UI.

---

**Local Setup**

1) **Clone & create a virtual environment**
```bash
git clone git clone https://huggingface.co/spaces/SameehK/MusRec
cd MusRec
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Environment variables**
Create a .env file in the project root: 
```bash
GROQ_API_KEY=your_groq_key
YOUTUBE_API_KEY=your_youtube_data_api_key
```
Tip: In Google Cloud Console, enable YouTube Data API v3 and restrict your key to that API.

4) **Run the app**
```bash
python musrec.py
```
Open the local URL (default http://127.0.0.1:7860).





