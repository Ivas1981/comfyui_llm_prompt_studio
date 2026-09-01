import sys, os, json, time, argparse, urllib.request

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # Testing package dir
import styles

SERVER = "http://localhost:1234/v1"
MODEL = None


def detect_loaded_model(server_base):
    # /api/v1/models returns only models currently loaded in LM Studio.
    api_root = server_base.rsplit("/v1", 1)[0]
    try:
        req = urllib.request.Request(
            api_root + "/api/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data") or []
        if models:
            return models[0]["id"]
    except Exception as e:
        sys.stderr.write("model auto-detect failed: %r\n" % (e,))
    return None

SCENARIOS = [
    {
        "name": "S1 photography / negative=on / natural / face=off (EN idea)",
        "preset": "portrait_photography",
        "nsfw": False, "prompt_format": "natural", "negative_prompt": True,
        "face_prompt": False, "blend_styles": "", "architecture": "sdxl",
        "idea": "a freckled woman laughing in a sunlit wheat field, 35mm",
    },
    {
        "name": "S2 anime / negative=off (no-negative) / tags / face=on (RU idea)",
        "preset": "anime",
        "nsfw": False, "prompt_format": "tags", "negative_prompt": False,
        "face_prompt": True, "blend_styles": "", "architecture": "pony",
        "idea": "спокойная девушка-самурай у реки на рассвете",
    },
    {
        "name": "S3 blend cinematic+oil_painting+cyberpunk / structured / negative=on",
        "preset": "cinematic",
        "nsfw": False, "prompt_format": "structured", "negative_prompt": True,
        "face_prompt": False, "blend_styles": "oil_painting,cyberpunk",
        "architecture": "sdxl",
        "idea": "a neon megacity skyline at dusk with flying cars",
    },
]


def call_lm(sys_text, idea):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": f"Idea: {idea}\n\nRespond ONLY with the JSON object (positive, negative, scene_name, face_positive, face_negative)."},
        ],
        "temperature": 0.7,
        "max_tokens": 600,
        "stream": False,
    }
    req = urllib.request.Request(
        SERVER + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


def extract_json(t):
    # gemma may emit special tokens (e.g. <unused24>) before the JSON object.
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return t[s:e + 1]
    return t


def main():
    parser = argparse.ArgumentParser(description="Live LM Studio prompt-generation test")
    parser.add_argument("--model", default=None, help="Override the auto-detected loaded model id")
    args = parser.parse_args()
    model = args.model or detect_loaded_model(SERVER)
    if not model:
        sys.stderr.write(
            "No model is loaded in LM Studio and --model was not given.\n"
            "Load a model in LM Studio, or run with --model <id>.\n"
        )
        sys.exit(1)
    global MODEL
    MODEL = model
    sys.stdout.write("[MODEL] using: %s\n" % MODEL)
    sys.stdout.flush()

    out_path = os.path.join(os.path.dirname(__file__), "live_results.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("[MODEL] %s\n" % MODEL)
        for sc in SCENARIOS:
            out.write("\n================ " + sc["name"] + " ================\n")
            sys.stdout.write("\n[RUN] " + sc["name"] + "\n")
            sys.stdout.flush()
            preset = styles.resolve_style_token(sc["preset"])
            sys_text = styles.build_system_prompt(
                preset,
                nsfw=sc["nsfw"], prompt_format=sc["prompt_format"],
                negative_prompt=sc["negative_prompt"], face_prompt=sc["face_prompt"],
                blend_styles=sc["blend_styles"], architecture=sc["architecture"],
            )
            out.write("--- SYSTEM PROMPT (first 1200 chars) ---\n")
            out.write(sys_text[:1200] + "\n")
            t0 = time.time()
            try:
                raw = call_lm(sys_text, sc["idea"])
            except Exception as e:
                out.write("ERROR: " + repr(e) + "\n")
                sys.stdout.write("[ERR] " + repr(e) + "\n")
                sys.stdout.flush()
                continue
            dt = time.time() - t0
            out.write(f"--- MODEL RAW ({dt:.1f}s) ---\n")
            out.write(raw + "\n")
            try:
                parsed = json.loads(extract_json(raw))
                out.write("--- PARSED ---\n")
                out.write(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
            except Exception as e:
                out.write("PARSE FAILED: " + repr(e) + "\n")
            out.flush()
    sys.stdout.write("\n[DONE] results -> tools/live_results.txt\n")


if __name__ == "__main__":
    main()
