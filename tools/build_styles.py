"""Generate the Styles/ folder from presets_default.json + a curated new catalog.

Run from the Testing repo root:
    python tools/build_styles.py

Produces:
    Styles/system_prompts.json   - base prompts + nsfw/prompt_format/negative fragments + architecture_guidance
    Styles/<direction>.json      - style presets (migrated 51 + curated new catalog entries)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "presets_default.json")
STYLES_DIR = os.path.join(ROOT, "Styles")
GEN_NEG = (
    "NO-NEGATIVE MODE: distilled models sample at CFG ~1 where the negative prompt is "
    "mathematically ignored, so you MUST express every stylistic and quality constraint as a "
    "POSITIVE statement inside the positive prompt. Explicitly name what must be present and "
    "phrase exclusions positively (e.g. 'sharp focus, clean lines, no blur, no extra limbs')."
)


def mk(id, name, desc, sp, pos, neg, blend, cat, sp_neg=None, disabled=False, extends=None):
    d = {
        "id": id, "name": name, "description": desc, "system_prompt": sp,
        "style_tags_positive": pos, "style_tags_negative": neg,
        "blend_note": blend, "category": cat,
        "disabled_in_no_negative_mode": disabled,
    }
    if sp_neg:
        d["system_prompt_no_negative"] = sp_neg
    if extends:
        d["extends"] = extends
    return d


# ---------------------------------------------------------------------------
# Base system prompts + fragments  ->  system_prompts.json
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "schema_version": "2.0.0",
    "writer_system": (
        "You are an expert prompt engineer for text-to-image diffusion models "
        "(SDXL, SD1.5, Flux, Pony, Illustrious). You convert the user's idea into a "
        "high-quality image-generation prompt.\n"
        "Language: write the ENTIRE prompt in English, regardless of the user's input "
        "language. Never mirror the user's language.\n"
        "Always respond with ONLY a single JSON object of this exact shape:\n"
        "{\n"
        '  "positive": "<the main image prompt>",\n'
        '  "negative": "<traits to avoid; empty string if none>",\n'
        '  "scene_name": "<short 2-4 word slug, lowercase, underscores>",\n'
        '  "face_positive": "<empty unless a face is clearly described>",\n'
        '  "face_negative": "<empty unless a face is clearly described>"\n'
        "}\n"
        "ALWAYS finish your reply with the complete JSON object."
    ),
    "face_instruction": (
        "FACE — on: Also include 'face_positive' (20-40 words focused ONLY on the face: "
        "apparent age and markers, skin tone, freckles/moles/scars, eye color and shape, lip "
        "shape, near-face hair, makeup; describe realistic skin texture in photographic terms; "
        "match the scene's lighting and style) and 'face_negative' (a short list of face flaws "
        "to avoid). Write fresh for each face; never reuse fixed phrases."
    ),
    "critic_system": (
        "You are a strict critic of generated images. Compare the image against the prompt and "
        "score 0-10 on prompt fidelity, anatomical accuracy, photorealism/style match, and "
        "framing/composition. Respond STRICTLY in JSON: "
        '{"score": <int 0-10>, "verdict": "<text>", "revision_notes": "<text>"}.'
    ),
    "describe": (
        "Describe this image in great detail in English: subject (gender, approximate age, body "
        "type, build, facial features, hairstyle, visible skin details), clothing and fabric "
        "texture, environment, time of day, lighting direction and quality, colors, composition, "
        "framing, camera perspective, style and mood. Write coherent prose, no lists. "
        "Language requirement: you MUST write the ENTIRE response in English."
    ),
    "composer": (
        "You are an expert in image prompts. Based on the scene description and the user's "
        "requested changes, compose a fresh, detailed prompt in English only. Layer it: "
        "1. Subject, 2. Action/Pose/framing, 3. Environment/textures, 4. Lighting, "
        "5. Camera/Optics. Rules: write connected sentences, do not reuse fixed templates, and "
        "stay in English."
    ),
    "reasoning_hint": (
        " If you use reasoning, keep it under 5 sentences and ALWAYS finish your reply with the "
        "complete JSON object."
    ),
    "nsfw_policy": {
        "on": (
            "SAFETY — nsfw permitted: Explicit or mature content is allowed ONLY when the user "
            "explicitly requests it; otherwise keep the prompt safe-for-work."
        ),
        "off": (
            "SAFETY — safe-for-work: Keep all content safe-for-work. Do not describe nudity, "
            "sexual acts, graphic violence, or explicit gore. If the request implies such "
            "content, redirect to a tasteful, non-explicit interpretation."
        ),
    },
    "prompt_format": {
        "natural": (
            "FORMAT — natural: Write the positive (and negative) as connected English sentences "
            "describing the scene in descending importance (most important first), single paragraph."
        ),
        "tags": (
            "FORMAT — tags: Write the positive as a comma-separated list of descriptive tags "
            "ordered by importance (most important first). No full sentences."
        ),
        "weighted": (
            "FORMAT — weighted: Write the positive as comma-separated tags; emphasize key terms "
            "with parentheses weights, e.g. (intricate details:1.3), (sharp focus:1.2)."
        ),
        "structured": (
            "FORMAT — structured: Write the positive as labeled blocks: "
            "[Subject] ... [Lighting] ... [Style] ... [Camera] ... [Composition] ..."
        ),
        "midjourney": (
            "FORMAT — midjourney: Write Midjourney-style: comma-separated descriptors with :: "
            "weights, and express exclusions with --no tokens when a negative field is unavailable."
        ),
        "booru": (
            "FORMAT — booru: Write booru-style comma-separated tokens; lead with quality/rating "
            "tags (score_9, score_8_up, ...), then subject, then descriptors. Avoid natural sentences."
        ),
    },
    "negative": {
        "on": (
            "NEGATIVE — required: Populate the 'negative' field with a comma-separated list of "
            "traits to avoid (e.g. blurry, low quality, extra limbs, watermark, text, deformed hands)."
        ),
        "off": (
            "NEGATIVE — off (self-contained positive): Leave the 'negative' field EMPTY. Express "
            "every constraint INSIDE the positive as a positive statement: explicitly name what must "
            "be present AND phrase exclusions within the positive text (e.g. 'sharp focus, clean "
            "lines, no blur, no extra limbs, no watermark'). Do not forbid things in a separate negative."
        ),
    },
    "face_prompt": {
        "on": "",  # base writer_system already requests face fields; face_instruction carries detail
        "off": (
            "FACE — off: Leave face_positive and face_negative empty unless a face is clearly "
            "described; do not force face details."
        ),
    },
}

# architecture_guidance is migrated verbatim from presets_default.json
with open(SRC, "r", encoding="utf-8") as f:
    _src = json.load(f)
SYSTEM_PROMPTS["architecture_guidance"] = _src.get("architecture_guidance", {})


# ---------------------------------------------------------------------------
# Migration map: preset id -> (direction file, group label)
# ---------------------------------------------------------------------------
FILE_FOR = {
    "photorealism": ("photography", "Photography"),
    "portrait_photography": ("photography", "Photography"),
    "boudoir_glamour": ("photography", "Photography"),
    "street_photography": ("photography", "Photography"),
    "fashion_editorial": ("photography", "Photography"),
    "macro_photography": ("photography", "Photography"),
    "landscape_photography": ("photography", "Photography"),
    "black_white": ("photography", "Photography"),
    "vintage_film": ("film_stocks", "Analog Film"),
    "cinematic": ("cinema", "Cinema"),
    "anime": ("animation", "Animation"),
    "comic": ("comics", "Comics & Manga"),
    "3d_render": ("render_engines", "Render Engines"),
    "digital_art": ("digital_painting", "Digital Painting"),
    "pixel_art": ("game_art", "Game Art"),
    "minimalist": ("modern_art", "Modern Art"),
    "vaporwave": ("period", "Period & Style"),
    "glitch_art": ("experimental", "Experimental"),
    "low_poly_3d": ("game_art", "Game Art"),
    "synthwave": ("period", "Period & Style"),
    "sticker_art": ("poster", "Poster & Graphic"),
    "oil_painting": ("painting", "Traditional Media"),
    "watercolor": ("painting", "Traditional Media"),
    "charcoal_sketch": ("painting", "Traditional Media"),
    "pastel": ("painting", "Traditional Media"),
    "ink_illustration": ("painting", "Traditional Media"),
    "collage": ("painting", "Traditional Media"),
    "art_nouveau": ("art_movements", "Art Movements"),
    "impressionism": ("art_movements", "Art Movements"),
    "surrealism": ("art_movements", "Art Movements"),
    "baroque": ("art_movements", "Art Movements"),
    "art_deco": ("art_movements", "Art Movements"),
    "pop_art": ("modern_art", "Modern Art"),
    "expressionism": ("art_movements", "Art Movements"),
    "ukiyoe": ("asian_art", "Asian Art"),
    "chinese_ink_painting": ("asian_art", "Asian Art"),
    "sumi_e": ("asian_art", "Asian Art"),
    "mahou_shoujo": ("animation", "Animation"),
    "manhwa_webtoon": ("comics", "Comics & Manga"),
    "fantasy": ("fantasy", "Fantasy"),
    "lovecraftian_horror": ("fantasy_horror", "Fantasy & Horror"),
    "gothic_horror": ("fantasy_horror", "Fantasy & Horror"),
    "dark_fantasy": ("fantasy_horror", "Fantasy & Horror"),
    "fairy_tale": ("fantasy", "Fantasy"),
    "cyberpunk": ("period", "Period & Style"),
    "steampunk": ("period", "Period & Style"),
    "art_brut_naive": ("modern_art", "Modern Art"),
    "retro_80s": ("period", "Period & Style"),
    "victorian": ("period", "Period & Style"),
    "noir": ("comics", "Comics & Manga"),
    "graffiti": ("street_tattoo", "Street Art & Tattoo"),
}

# ---------------------------------------------------------------------------
# Curated NEW catalog presets (style_prompt concise; no-negative auto-generated)
# ---------------------------------------------------------------------------
NEW = {
    "cinema": [
        mk("wes_anderson", "Wes Anderson", "Symmetric, pastel-palette idiosyncratic film framing",
           "Style focus: Wes Anderson cinema. Compose with rigorous central symmetry, flat even "
           "lighting, a curated pastel and saturated color palette, and deadpan storybook staging; "
           "name specific props and costume colors.",
           ["wes anderson style", "symmetric composition", "pastel palette", "flat lighting", "storybook staging"],
           ["asymmetric", "messy", "dark moody"],
           "symmetry, pastel palette, deadpan staging", "Cinema"),
        mk("blockbuster", "Summer Blockbuster", "High-concept Hollywood spectacle framing",
           "Style focus: summer-blockbuster spectacle. Emphasize epic scale, dramatic hero lighting, "
           "explosive atmosphere, and widescreen composition with foreground hero subject.",
           ["cinematic", "epic scale", "dramatic hero lighting", "widescreen"],
           ["static", "flat"], "epic scale, hero lighting", "Cinema"),
    ],
    "animation": [
        mk("shonen", "Shonen Anime", "Energetic battle-anime look",
           "Style focus: shonen anime. Bold dynamic linework, high-contrast shading, speed lines, "
           "exaggerated expressive faces, and energetic action posing; use anime vocabulary.",
           ["shonen anime", "bold linework", "speed lines", "dynamic pose"],
           ["realistic", "photographic", "western comic"],
           "dynamic action, bold linework", "Animation"),
        mk("shojo", "Shojo Anime", "Soft decorative romance-anime look",
           "Style focus: shojo anime. Soft pastel colors, sparkly highlights, large expressive eyes, "
           "flowing hair, decorative backgrounds and gentle romantic mood.",
           ["shojo anime", "soft pastel", "sparkly highlights", "flowing hair"],
           ["gritty", "realistic"], "soft pastel, decorative", "Animation"),
        mk("disney_revival", "Western Animated Film", "Modern Western feature-animation look",
           "Style focus: modern Western animated feature. Appealing stylized characters, rounded "
           "forms, rich painterly backgrounds, expressive cartoon acting, warm lighting.",
           ["western animation", "stylized characters", "painterly background"],
           ["anime", "realistic", "rough"],
           "stylized, painterly", "Animation"),
    ],
    "comics": [
        mk("ligne_claire", "Ligne Claire", "Hergé-style clean-line illustration",
           "Style focus: ligne claire. Uniform clean contours, no shading gradients, flat even color "
           "areas, precise detail and a calm objective composition.",
           ["ligne claire", "clean linework", "flat colors", "uniform contours"],
           ["messy lines", "heavy shading", "gritty"], "clean uniform linework", "Comics & Manga"),
        mk("superhero", "Superhero Comic", "American superhero comic look",
           "Style focus: American superhero comic. Bold inked linework, saturated color, dramatic "
           "foreshortening, heroic musculature, dynamic cape and motion, halftone shading.",
           ["superhero comic", "bold ink", "saturated color", "dynamic pose"],
           ["realistic", "muted"], "bold ink, dynamic heroism", "Comics & Manga"),
        mk("bande_dessinee", "Bande Dessinee", "European album comic look",
           "Style focus: European bande dessinee. Precise clean pen work, realistic proportions, "
           "rich detailed settings, flat color with subtle modeling, witty composition.",
           ["bande dessinee", "clean pen", "detailed setting", "flat color"],
           ["messy", "anime"], "precise clean pen", "Comics & Manga"),
        mk("indie_graphic_novel", "Indie Graphic Novel", "Literary alternative comics look",
           "Style focus: indie graphic novel. Personal ink or watercolor texture, irregular hand-drawn "
           "line, muted earthy palette, intimate framing, narrative quiet mood.",
           ["indie comic", "hand-drawn", "muted palette", "ink texture"],
           ["glossy", "superhero"], "hand-drawn, intimate", "Comics & Manga"),
    ],
    "render_engines": [
        mk("unreal_engine", "Unreal Engine 5", "Real-time UE5 render look",
           "Style focus: Unreal Engine 5 real-time render. Nanite detail, Lumen global illumination, "
           "soft indirect light, crisp PBR materials, subtle bloom, game-cinematic framing.",
           ["unreal engine 5", "nanite", "lumen gi", "pbr materials"],
           ["flat", "painterly"], "real-time PBR, Lumen GI", "Render Engines"),
        mk("octane", "Octane Render", "Octane GPU render look",
           "Style focus: Octane render. Physically based lighting, rich volumetrics, glossy reflective "
           "surfaces, cinematic color, smooth denoised clean output.",
           ["octane render", "pbr", "volumetric", "glossy"],
           ["noisy", "flat"], "PBR, volumetrics", "Render Engines"),
        mk("blender_cycles", "Blender Cycles", "Cycles path-traced look",
           "Style focus: Blender Cycles. Path-traced global illumination, soft caustics, neutral PBR "
           "shading, clean studio or HDRI lighting, subtle filmic tone.",
           ["blender cycles", "path tracing", "pbr", "hdri lighting"],
           ["painterly"], "path-traced PBR", "Render Engines"),
        mk("redshift", "Redshift", "Redshift production render look",
           "Style focus: Redshift. Production CGI look, tight specular highlights, deep shadows, "
           "saturated cinematic grade, fine motion-blur-ready detail.",
           ["redshift render", "cinematic cgi", "specular"],
           ["flat"], "production CGI", "Render Engines"),
    ],
    "digital_painting": [
        mk("concept_art", "Concept Art", "Professional environment concept painting",
           "Style focus: production concept art. Bold readable silhouette, atmospheric perspective, "
           "loose confident brushwork, cinematic key light, painterly but legible detail.",
           ["concept art", "atmospheric", "painterly", "cinematic key light"],
           ["photographic", "flat"], "painterly, atmospheric", "Digital Painting"),
        mk("matte_painting", "Matte Painting", "Photoreal digital environment painting",
           "Style focus: digital matte painting. Seamless photoreal environments, fine detail at "
           "infinity, layered atmosphere, invisible brushwork blending real and painted elements.",
           ["matte painting", "photoreal environment", "atmospheric depth"],
           ["visible brush", "flat"], "photoreal environment", "Digital Painting"),
        mk("digital_illustration", "Digital Illustration", "Modern vector-ish illustration",
           "Style focus: modern digital illustration. Clean shapes, controlled palette, crisp edges, "
           "stylized but polished, suitable for editorial or game key art.",
           ["digital illustration", "clean shapes", "stylized"],
           ["messy", "photoreal"], "clean stylized", "Digital Painting"),
    ],
    "game_art": [
        mk("cel_shaded_3d", "Cel-Shaded 3D", "Toon-shaded 3D game look",
           "Style focus: cel-shaded 3D. Hard rim light, flat toon gradient, bold outline, stylized "
           "forms, bright readable palette like a 3D animated game.",
           ["cel shaded", "toon shading", "bold outline", "stylized 3d"],
           ["realistic", "photographic"], "toon shading, outline", "Game Art"),
        mk("photorealistic_aaa", "Photorealistic AAA", "Next-gen game render look",
           "Style focus: photorealistic AAA game render. High-poly detail, PBR materials, real-time "
           "GI, subtle film grain, hero-asset lighting, screenshot-from-a-game framing.",
           ["aaa game render", "pbr", "photoreal", "realtime gi"],
           ["flat", "toon"], "PBR game render", "Game Art"),
        mk("isometric", "Isometric Game", "Isometric game-art look",
           "Style focus: isometric game art. Fixed 30-degree axonometric view, readable tile-based "
           "forms, clean stylized shading, cohesive palette, diorama-like clarity.",
           ["isometric", "axonometric", "stylized", "tile based"],
           ["perspective", "messy"], "isometric clarity", "Game Art"),
        mk("hand_drawn_2d", "Hand-Drawn 2D", "2D game animation look",
           "Style focus: hand-drawn 2D game art. Slightly imperfect ink line, frame-by-frame "
           "personality, limited but charming palette, paper or cel texture.",
           ["hand drawn 2d", "ink line", "cel texture"],
           ["3d", "photoreal"], "hand-drawn charm", "Game Art"),
    ],
    "modern_art": [
        mk("op_art", "Op Art", "Optical illusion geometric art",
           "Style focus: Op Art. Precise black-and-white (or two-color) geometric patterns that "
           "create moire, vibration and illusion of movement; mathematically regular.",
           ["op art", "geometric pattern", "moire", "optical illusion"],
           ["organic", "messy", "painterly"], "geometric optical", "Modern Art"),
        mk("constructivism", "Constructivism", "Soviet avant-garde design art",
           "Style focus: Russian Constructivism. Bold diagonal compositions, photomontage, red/black "
           "palette, stencil typography, industrial and revolutionary iconography.",
           ["constructivism", "diagonal", "photomontage", "red black"],
           ["soft", "pastel"], "bold diagonal, red/black", "Modern Art"),
        mk("de_stijl", "De Stijl", "Mondrian neo-plasticism",
           "Style focus: De Stijl / Neo-plasticism. Strict primary-color blocks separated by thick "
           "black grid lines on white, pure horizontal/vertical geometry, total abstraction.",
           ["de stijl", "mondrian", "primary colors", "black grid"],
           ["curves", "realism"], "grid, primary colors", "Modern Art"),
        mk("abstract_expressionism", "Abstract Expressionism", "Action-painting abstraction",
           "Style focus: Abstract Expressionism. Spontaneous gestural brushstrokes, drips, large "
           "all-over composition, emotional color, visible physical paint energy.",
           ["abstract expressionism", "gestural", "drip", "expressive color"],
           ["figurative", "clean"], "gestural abstraction", "Modern Art"),
        mk("color_field", "Color Field", "Rothko-style field painting",
           "Style focus: Color Field painting. Large soft-edged areas of luminous color, minimal "
           "form, meditative atmosphere, subtle tonal bleed at edges.",
           ["color field", "soft edges", "luminous color"],
           ["detail", "linework"], "soft color fields", "Modern Art"),
    ],
    "experimental": [
        mk("pixel_sorting", "Pixel Sorting", "Datamosh pixel-sort glitch",
           "Style focus: pixel sorting. Rows/columns of pixels reordered by brightness creating "
           "streaked glitch rivers, abstract corruption, digital decay over the image.",
           ["pixel sorting", "glitch", "datamosh", "streaked"],
           ["clean", "smooth"], "glitch streaks", "Experimental"),
        mk("long_exposure", "Long Exposure", "Motion-blur light photography",
           "Style focus: long-exposure photography. Smooth silky motion blur of water/cloud/light, "
           "trails of moving lights, calm ethereal atmosphere, tripod stillness.",
           ["long exposure", "motion blur", "light trails", "silky"],
           ["noisy", "static harsh"], "silky motion blur", "Experimental"),
        mk("double_exposure", "Double Exposure", "Overlay blend photography",
           "Style focus: double exposure. Two images blended by luminosity — a silhouette filled "
           "with a contrasting scene, dreamlike conceptual overlap.",
           ["double exposure", "overlay", "silhouette blend"],
           ["flat single"], "conceptual overlay", "Experimental"),
        mk("kaleidoscope", "Kaleidoscope", "Mirror-symmetric pattern art",
           "Style focus: kaleidoscope. Radial mirror symmetry, repeated fractured segments, saturated "
           "pattern, hypnotic ornamental geometry.",
           ["kaleidoscope", "mirror symmetry", "radial pattern"],
           ["asymmetry"], "radial symmetry", "Experimental"),
        mk("chromatic_aberration", "Chromatic Aberration", "RGB-split optical glitch",
           "Style focus: chromatic aberration. Deliberate red/blue channel split at edges, lens "
           "fringing, glitchy optical distortion, digital imperfection.",
           ["chromatic aberration", "rgb split", "lens fringing"],
           ["perfect optics"], "rgb split", "Experimental"),
    ],
    "poster": [
        mk("swiss_poster", "Swiss Style Poster", "International-typographic poster",
           "Style focus: Swiss/International poster. Grid-based layout, Helvetica-style sans, "
           "asymmetric balance, generous whitespace, single bold accent color, objective clarity.",
           ["swiss style", "grid layout", "sans serif", "minimal"],
           ["ornate", "messy"], "grid, minimal type", "Poster & Graphic"),
        mk("wpa_poster", "WPA Poster", "Depression-era public poster",
           "Style focus: WPA poster. Silkscreen flat color, strong simple shapes, optimistic "
           "public-works iconography, limited earthy palette, bold readable composition.",
           ["wpa poster", "silkscreen", "flat color", "bold shapes"],
           ["photoreal", "busy"], "flat silkscreen", "Poster & Graphic"),
        mk("soviet_poster", "Soviet Poster", "Agitprop constructivist poster",
           "Style focus: Soviet agitprop poster. Heroic stylized figure, raised dynamic pose, "
           "bold red/black palette, diagonal energy, stark propaganda typography.",
           ["soviet poster", "agitprop", "heroic", "red black"],
           ["soft", "pastel"], "heroic red/black", "Poster & Graphic"),
        mk("travel_poster", "Travel Poster", "Mid-century travel advert",
           "Style focus: mid-century travel poster. Layered flat scenery, sunburst or atomic "
           "motifs, warm limited palette, optimistic stylization, clean vector shapes.",
           ["travel poster", "mid century", "flat scenery", "warm palette"],
           ["photoreal", "gritty"], "flat scenic", "Poster & Graphic"),
        mk("psychedelic_poster", "Psychedelic Poster", "60s liquid poster art",
           "Style focus: psychedelic poster. Swirling organic letterforms, melting gradients, "
           "vivid clashing colors, Art-Nouveau-influenced line, hallucinatory energy.",
           ["psychedelic", "swirling", "vivid gradient", "organic line"],
           ["minimal", "flat"], "swirling vivid", "Poster & Graphic"),
        mk("gig_poster", "Gig Poster", "Screenprinted music poster",
           "Style focus: gig poster. Hand-pulled screenprint texture, limited spot colors, bold "
           "illustrative focal image, slight off-register charm, music-scene attitude.",
           ["gig poster", "screenprint", "spot color", "bold illustration"],
           ["photoreal"], "screenprint texture", "Poster & Graphic"),
    ],
    "painting": [
        mk("classical_realism", "Classical Realism", "Academic representational painting",
           "Style focus: classical realist painting. Careful academic draftsmanship, smooth layered "
           "glazes, balanced composition, subtle naturalistic light, refined finish.",
           ["classical realism", "academic", "glazes", "naturalistic"],
           ["cartoon", "flat"], "academic finish", "Traditional Media"),
        mk("impasto_oil", "Impasto Oil", "Thick-textured油画 brushwork",
           "Style focus: impasto oil painting. Thick sculpted paint ridges, visible bristle "
           "direction, rich layered color, tactile surface catching light.",
           ["impasto", "thick paint", "brush texture", "oil"],
           ["smooth", "flat"], "thick paint texture", "Traditional Media"),
        mk("gothic_illumination", "Manuscript Illumination", "Medieval gold-leaf art",
           "Style focus: medieval manuscript illumination. Gold-leaf ground, fine tempera detail, "
           "decorative borders, flattened perspective, jewel-like saturated color.",
           ["illumination", "gold leaf", "tempera", "ornate border"],
           ["modern", "photoreal"], "gold leaf, ornate", "Traditional Media"),
    ],
    "art_movements": [
        mk("cubism", "Cubism", "Analytic/Synthetic Cubist art",
           "Style focus: Cubism. Fragmented facets, multiple simultaneous viewpoints, geometric "
           "planes, muted analytic palette or collage synthesis, abstraction of the subject.",
           ["cubism", "facets", "geometric planes", "multiple viewpoints"],
           ["realistic", "smooth"], "faceted abstraction", "Art Movements"),
        mk("romanticism", "Romanticism", "19th-c sublime painting",
           "Style focus: Romanticism. Dramatic sublime nature, intense emotion, dynamic storms and "
           "ruins, expressive brushwork, luminous contrasts of light and dark.",
           ["romanticism", "sublime", "dramatic nature", "expressive"],
           ["clinical", "flat"], "sublime drama", "Art Movements"),
        mk("futurism_art", "Futurism", "Italian futurist art",
           "Style focus: Futurism. Celebration of speed and machine, lines of force, fragmented "
           "motion, dynamic diagonal energy, vigorous brush and industrial subject.",
           ["futurism", "lines of force", "motion", "dynamic diagonal"],
           ["static", "calm"], "dynamic motion", "Art Movements"),
        mk("dada", "Dada", "Anti-art absurd collage",
           "Style focus: Dada. Deliberate absurdity, photomontage, chance juxtaposition, "
           "anti-aesthetic typography, provocative nonsense and collage rupture.",
           ["dada", "photomontage", "absurd", "collage"],
           ["harmonious", "realistic"], "absurd collage", "Art Movements"),
    ],
    "asian_art": [
        mk("tang_figurative", "Tang Figurative", "Tang-dynasty painting look",
           "Style focus: Tang-dynasty figure painting. Elegant flowing robes, confident outline "
           "with light color wash, courtly posture, decorative patterning, warm mineral pigments.",
           ["tang dynasty", "flowing robe", "outline wash", "mineral pigment"],
           ["western", "photoreal"], "elegant outline wash", "Asian Art"),
        mk("persian_miniature_art", "Persian Miniature", "Persian manuscript painting",
           "Style focus: Persian miniature. Fine detailed flat perspective, jewel-like palette, "
           "dense ornamental flora, gold skies, narrative intimacy, precise outline.",
           ["persian miniature", "flat perspective", "ornamental", "gold sky"],
           ["western depth", "photoreal"], "jewel-like flat", "Asian Art"),
    ],
    "fantasy": [
        mk("high_fantasy", "High Fantasy", "Epic Tolkienesque fantasy art",
           "Style focus: high fantasy illustration. Sweeping secondary-world landscapes, armored "
           "heroes, detailed medievalism, dramatic epic light, rich painterly narrative detail.",
           ["high fantasy", "epic", "medieval", "painterly"],
           ["modern", "sci-fi"], "epic medieval", "Fantasy"),
        mk("mythic_creature", "Mythic Creature Art", "Dragon/beast fantasy painting",
           "Style focus: mythic-creature fantasy. Imposing dragons/beasts with believable anatomy, "
           "atmospheric scale, dramatic creature lighting, detailed scales and environment.",
           ["mythic creature", "dragon", "dramatic scale", "detailed"],
           ["cute", "flat"], "imposing creature", "Fantasy"),
    ],
    "fantasy_horror": [
        mk("eldritch", "Eldritch Horror", "Cosmic unknowable horror art",
           "Style focus: eldritch/cosmic horror. Non-Euclidean geometry, impossible vast creatures, "
           "sickly otherworldly palette, dread and insignificance of scale, eerie light.",
           ["eldritch", "cosmic horror", "non euclidean", "dread"],
           ["bright", "cheerful"], "cosmic dread", "Fantasy & Horror"),
        mk("gothic_architecture_horror", "Gothic Horror", "Gothic ruin horror art",
           "Style focus: Gothic horror. Crumbling cathedrals, mist, candelit shadow, pallid "
           "figures, Victorian decay, melancholic chiaroscuro.",
           ["gothic horror", "cathedral", "mist", "chiaroscuro"],
           ["bright", "modern"], "gothic decay", "Fantasy & Horror"),
    ],
    "period": [
        mk("mid_century_modern", "Mid-Century Modern", "50s-60s design aesthetic",
           "Style focus: mid-century modern. Clean organic-meets-geometric forms, warm wood and "
           "mustard/teal palette, optimistic retro-futurism, simple elegant styling.",
           ["mid century modern", "retro", "warm wood", "mustard teal"],
           ["gritty", "ornate"], "warm retro", "Period & Style"),
        mk("roaring_twenties", "Roaring Twenties", "1920s art-deco glamour",
           "Style focus: Roaring Twenties. Art-Deco glamour, flappers, jazz-age gold and black, "
           "geometric luxe, confident optimistic deco line.",
           ["roaring twenties", "art deco", "gold black", "flapper"],
           ["modern", "drab"], "deco glamour", "Period & Style"),
        mk("wild_west", "Wild West", "American frontier look",
           "Style focus: Wild West. Dusty frontier towns, leather and denim, sepia sunlight, "
           "wide desert horizons, weathered Americana.",
           ["wild west", "frontier", "dusty", "sepia"],
           ["urban", "sci-fi"], "dusty frontier", "Period & Style"),
    ],
    "street_tattoo": [
        mk("traditional_tattoo", "Traditional Tattoo", "Old-school tattoo flash",
           "Style focus: traditional (old-school) tattoo. Bold black outlines, limited saturated "
           "palette, iconic subject (anchor, rose, eagle), flat shading, clean flash composition.",
           ["traditional tattoo", "bold outline", "old school", "flash"],
           ["realism", "watercolor"], "bold old-school", "Street Art & Tattoo"),
        mk("irezumi", "Irezumi", "Japanese full-body tattoo",
           "Style focus: Irezumi. Japanese tattoo, flowing muscular integration of dragons/kinshi "
           "with waves and clouds, tebori texture, restricted traditional palette.",
           ["irezumi", "japanese tattoo", "dragon", "waves"],
           ["western", "neotrad"], "japanese flow", "Street Art & Tattoo"),
        mk("blackwork_tattoo", "Blackwork Tattoo", "Solid-black tattoo style",
           "Style focus: blackwork tattoo. Heavy solid black, negative-space line, occult or "
           "geometric motifs, high contrast, ritualistic symmetry.",
           ["blackwork tattoo", "solid black", "negative space", "geometric"],
           ["color", "soft"], "solid black", "Street Art & Tattoo"),
        mk("neo_traditional_tattoo", "Neo-Traditional", "Modern tattoo style",
           "Style focus: neo-traditional tattoo. Traditional boldness with richer gradient shading, "
           "expanded palette, dimensional forms, illustrative depth.",
           ["neo traditional tattoo", "bold outline", "gradient shading", "illustrative"],
           ["flat old school"], "bold + depth", "Street Art & Tattoo"),
        mk("banksy_stencil", "Banksy Stencil", "Street stencil art",
           "Style focus: Banksy-style stencil graffiti. High-contrast black silhouette on wall "
           "texture, concise satirical scene, spray-paint edge, urban location.",
           ["banksy", "stencil", "street art", "silhouette"],
           ["colorful tags", "messy"], "stencil satire", "Street Art & Tattoo"),
    ],
    "scifi": [
        mk("syd_mead", "Syd Mead", "Linear-blend hard sci-fi illustration",
           "Style focus: Syd Mead. Sleek hard-surface industrial design, reflective chrome and "
           "glass, warm hazy future cities, precise linear perspective, optimistic yet gritty.",
           ["syd mead", "hard sci-fi", "chrome", "future city"],
           ["fantasy", "medieval"], "sleek hard-surface", "Sci-Fi"),
        mk("ralph_mcquarrie", "Ralph McQuarrie", "Classic Star Wars concept look",
           "Style focus: Ralph McQuarrie. Cinematic sci-fi concept painting, atmospheric haze, "
           "dramatic rim light, iconic costume and vehicle silhouettes, painterly mood.",
           ["ralph mcquarrie", "sci-fi concept", "atmospheric", "rim light"],
           ["flat", "anime"], "cinematic sci-fi", "Sci-Fi"),
        mk("chris_foss", "Chris Foss", "Psychedelic ships illustration",
           "Style focus: Chris Foss. Gigantic detailed spacecraft, lush airbrushed gradients, "
           "saturated otherworldly color, epic scale, gleaming hulls.",
           ["chris foss", "spaceship", "airbrush", "epic scale"],
           ["gritty", "flat"], "airbrushed ships", "Sci-Fi"),
        mk("john_berkey", "John Berkey", "Majestic planetary sci-fi",
           "Style focus: John Berkey. Majestic planets and monuments, bold confident brush, "
           "dramatic scale, rich textured color, awe and grandeur.",
           ["john berkey", "planetary", "majestic", "grand scale"],
           ["small", "flat"], "majestic scale", "Sci-Fi"),
        mk("space_opera", "Space Opera", "Epic galactic sci-fi look",
           "Style focus: space opera. Sweeping galactic vistas, heroic crews, glowing tech, "
           "nebulae and capital ships, epic romantic lighting.",
           ["space opera", "galactic", "nebula", "epic"],
           ["drab", "mundane"], "epic galactic", "Sci-Fi"),
        mk("mecha", "Mecha", "Giant robot sci-fi look",
           "Style focus: mecha. Detailed giant robots, mechanical panel lines, cockpit and weapon "
           "detail, dynamic battle posing, industrially believable design.",
           ["mecha", "giant robot", "mechanical detail", "battle"],
           ["organic", "cute"], "mechanical detail", "Sci-Fi"),
        mk("retro_futurism", "Retro-Futurism", "Atomic-age future look",
           "Style focus: retro-futurism. 1950s vision of the future — rocket fins, googie "
           "architecture, pastel Jetsons optimism, analog space-age kitsch.",
           ["retro futurism", "atomic age", "googie", "pastel future"],
           ["grim", "modern"], "atomic optimism", "Sci-Fi"),
    ],
    "nature_events": [
        mk("storm", "Storm", "Dramatic tempest photography",
           "Style focus: storm photography. Towering thunderheads, rain shafts, lightning, "
           "dramatic chiaroscuro sky, sense of awe and scale.",
           ["storm", "thunderhead", "lightning", "dramatic sky"],
           ["calm", "flat light"], "dramatic sky", "Natural Events"),
        mk("aurora_borealis", "Aurora Borealis", "Northern lights photography",
           "Style focus: aurora photography. Shimmering green/violet curtains over dark landscape, "
           "long exposure, reflection in water, silent cold majesty.",
           ["aurora borealis", "northern lights", "long exposure", "reflection"],
           ["daylight", "busy"], "shimmering curtains", "Natural Events"),
        mk("volcanic_eruption", "Volcanic Eruption", "Eruption photography",
           "Style focus: volcanic eruption. Glowing lava, ash plume, incandescent flows, stark "
           "contrast of fire and dark rock, primordial drama.",
           ["volcanic eruption", "lava", "ash plume", "glow"],
           ["calm", "pastel"], "fire vs rock", "Natural Events"),
        mk("tornado", "Tornado / Supercell", "Severe storm photography",
           "Style focus: supercell/tornado. Rotation, green-tinged sky, funnel to ground, "
           "ominous mammatus clouds, vast prairie scale.",
           ["tornado", "supercell", "funnel", "green sky"],
           ["calm"], "ominous rotation", "Natural Events"),
        mk("nat_geo_doc", "Nat Geo Documentary", "Editorial wildlife photography",
           "Style focus: National Geographic documentary. Pristine wildlife in habitat, "
           "scientific accuracy, golden-hour light, unobtrusive telephoto realism.",
           ["national geographic", "wildlife", "golden hour", "realistic"],
           ["staged", "cartoon"], "pristine realism", "Natural Events"),
        mk("landscape_photo", "Landscape Photography", "Scenic landscape photo",
           "Style focus: landscape photography. Expansive vistas, layered depth, balanced "
           "composition, atmospheric light, fine natural detail.",
           ["landscape photography", "vista", "atmospheric", "balanced"],
           ["cluttered", "flat"], "expansive vista", "Natural Events"),
        mk("extreme_weather", "Extreme Weather", "Severe weather photography",
           "Style focus: extreme weather. Blizzards, floods, heat haze, monumental atmosphere, "
           "human scale against nature's force, raw documentary intensity.",
           ["extreme weather", "blizzard", "flood", "monumental"],
           ["calm", "soft"], "nature's force", "Natural Events"),
    ],
    "illustration": [
        mk("norman_rockwell", "Norman Rockwell", "American narrative illustration",
           "Style focus: Norman Rockwell. Warm narrative realism, relatable everyday Americana, "
           "precise storytelling detail, gentle humor, luminous indoor light.",
           ["norman rockwell", "narrative realism", "americana", "storytelling"],
           ["abstract", "gritty"], "warm narrative", "Illustration"),
        mk("alphonse_mucha", "Alphonse Mucha", "Art-Nouveau poster illustration",
           "Style focus: Alphonse Mucha. Elegant women in ornate arches, flowing hair, decorative "
           "botanical frames, muted jewel palette, flat idealized beauty.",
           ["alphonse mucha", "art nouveau", "ornate frame", "flowing hair"],
           ["hard", "modern"], "ornate elegance", "Illustration"),
        mk("childrens_book", "Children's Book", "Picture-book illustration",
           "Style focus: children's-book illustration. Charming stylized characters, soft warm "
           "palette, friendly simplified forms, whimsical storybook detail.",
           ["childrens book", "charming", "soft palette", "whimsical"],
           ["gritty", "realistic"], "charming whimsy", "Illustration"),
        mk("pulp_cover", "Pulp Cover", "Pulp-magazine illustration",
           "Style focus: pulp-magazine cover. Dramatic action, lurid saturated color, daring "
           "pose, bold simplified forms, sensational mood.",
           ["pulp cover", "dramatic", "saturated", "bold"],
           ["soft", "muted"], "sensational drama", "Illustration"),
        mk("vintage_advertising", "Vintage Advertising", "Retro ad illustration",
           "Style focus: vintage advertising illustration. Clean mid-century product hero, limited "
           "palette, confident flat shading, optimistic commercial charm.",
           ["vintage advertising", "retro ad", "flat shading", "product hero"],
           ["photoreal", "messy"], "retro commercial", "Illustration"),
        mk("editorial_illustration", "Editorial Illustration", "Magazine opinion illustration",
           "Style focus: editorial illustration. Conceptual metaphor, strong graphic idea, "
           "controlled palette, stylized but intelligent, fits a printed column.",
           ["editorial illustration", "conceptual", "graphic", "stylized"],
           ["literal", "photoreal"], "conceptual idea", "Illustration"),
    ],
    "architecture": [
        mk("brutalism", "Brutalism", "Raw concrete architecture",
           "Style focus: Brutalist architecture. Raw board-marked concrete, monumental massing, "
           "repetitive geometric modules, stark shadows, imposing honesty of material.",
           ["brutalism", "concrete", "monolithic", "geometric"],
           ["ornate", "soft"], "raw concrete mass", "Architecture"),
        mk("gothic_architecture", "Gothic Architecture", "Cathedral architecture",
           "Style focus: Gothic architecture. Pointed arches, ribbed vaults, soaring stained "
           "glass, flying buttresses, vertical aspiration and ornament.",
           ["gothic architecture", "pointed arch", "stained glass", "vaulted"],
           ["modern", "plain"], "vertical ornament", "Architecture"),
        mk("art_deco_arch", "Art Deco Architecture", "Deco building design",
           "Style focus: Art Deco architecture. Stepped ziggurat forms, streamlined vertical "
           "emphasis, chrome and geometric luxe, optimistic machine-age elegance.",
           ["art deco architecture", "ziggurat", "streamlined", "geometric luxe"],
           ["rough", "organic"], "stepped luxe", "Architecture"),
        mk("bauhaus", "Bauhaus", "Functional modernist design",
           "Style focus: Bauhaus architecture. White cubic volumes, ribbon windows, primary "
           "color accents, functional honesty, asymmetry and clean geometry.",
           ["bauhaus", "white cube", "ribbon windows", "functional"],
           ["ornate", "historical"], "white cubic", "Architecture"),
        mk("victorian_arch", "Victorian Architecture", "Gothic-revival building",
           "Style focus: Victorian architecture. Ornate facades, bay windows, turrets, patterned "
           "brick, elaborate trim, cluttered romantic detail.",
           ["victorian architecture", "ornate facade", "turret", "trim"],
           ["minimal", "modern"], "ornate detail", "Architecture"),
        mk("mid_century_arch", "Mid-Century Architecture", "Modernist house design",
           "Style focus: mid-century modern architecture. Flat or butterfly roofs, floor-to-ceiling "
           "glass, indoor-outdoor flow, warm wood and stone, optimistic horizontality.",
           ["mid century architecture", "flat roof", "glass", "warm wood"],
           ["heavy", "ornate"], "horizontal glass", "Architecture"),
        mk("postmodern_arch", "Postmodern Architecture", "Playful building design",
           "Style focus: postmodern architecture. Ironic historical quotation, bold color, "
           "playful forms, columns used decoratively, anti-purity exuberance.",
           ["postmodern architecture", "playful", "bold color", "quotation"],
           ["austere", "pure"], "playful quotation", "Architecture"),
        mk("deconstructivism", "Deconstructivism", "Fragmented building design",
           "Style focus: Deconstructivist architecture. Fragmented non-rectilinear forms, "
           "warped surfaces, deliberate dissonance, sculptural dynamic massing.",
           ["deconstructivism", "fragmented", "warped", "sculptural"],
           ["symmetrical", "calm"], "fragmented mass", "Architecture"),
    ],
    "film_stocks": [
        mk("kodachrome", "Kodachrome", "Classic warm slide film",
           "Style focus: Kodachrome film. Rich warm skin tones, deep saturated reds/blues, fine "
           "grain, classic documentary realism.",
           ["kodachrome", "warm tones", "saturated", "fine grain"],
           ["cool", "digital clean"], "warm saturated", "Analog Film"),
        mk("portra_400", "Portra 400", "Soft pastel portrait film",
           "Style focus: Portra 400 film. Soft pastel palette, gentle skin rendition, smooth "
           "grain, flattering low-contrast portraiture.",
           ["portra 400", "soft pastel", "smooth grain", "flattering"],
           ["harsh", "oversaturated"], "soft pastel", "Analog Film"),
        mk("ektachrome", "Ektachrome", "Vivid slide film",
           "Style focus: Ektachrome film. Crisp vivid color, cooler bias, clean fine grain, "
           "punchy travel/documentary look.",
           ["ektachrome", "vivid", "cool", "fine grain"],
           ["muted", "soft"], "crisp vivid", "Analog Film"),
        mk("cinestill_800t", "Cinestill 800T", "Tungsten night film",
           "Style focus: Cinestill 800T. Tungsten-balanced night photography, glowing red halo "
           "around highlights, punchy contrast, cinematic low-light.",
           ["cinestill 800t", "tungsten", "red halo", "night"],
           ["daylight", "flat"], "halo night", "Analog Film"),
        mk("polaroid", "Polaroid", "Instant photo look",
           "Style focus: Polaroid instant photo. Soft dreamy contrast, faded muted color, white "
           "frame border, nostalgic imperfection, slight vignette.",
           ["polaroid", "instant photo", "faded", "white border"],
           ["sharp", "hi-fi"], "faded nostalgia", "Analog Film"),
        mk("lomo", "Lomo", "Lo-fi saturated film",
           "Style focus: Lomography. Heavy saturated color, wild vignetting, unpredictable "
           "exposure, lo-fi plastic-lens charm, happy accidents.",
           ["lomo", "saturated", "vignette", "lo-fi"],
           ["clean", "clinical"], "lo-fi saturation", "Analog Film"),
        mk("infrared_film", "Infrared Film", "False-color IR photo",
           "Style focus: infrared film. Foliage glows white, deep black skies, dreamlike false "
           "color, ethereal otherworldly calm.",
           ["infrared film", "false color", "glowing foliage", "black sky"],
           ["normal color", "realistic"], "false-color dream", "Analog Film"),
        mk("cross_process", "Cross-Processing", "X-pro film look",
           "Style focus: cross-processed film. Shifted color (teal shadows, acidic highlights), "
           "high contrast, unpredictable dye shift, edgy lo-fi character.",
           ["cross processing", "xpro", "shifted color", "high contrast"],
           ["natural", "smooth"], "acidic shift", "Analog Film"),
    ],
    "craft_3d": [
        mk("claymation", "Claymation", "Stop-motion clay look",
           "Style focus: claymation. Hand-made clay textures, visible fingerprints, slightly "
           "imperfect forms, charming stop-motion lighting, tactile warmth.",
           ["claymation", "clay texture", "handmade", "stop motion"],
           ["smooth cgi", "photoreal"], "clay texture", "Craft & Physical"),
        mk("papercraft", "Papercraft", "Cut-paper diorama look",
           "Style focus: papercraft. Layered cut paper, visible fold and cut edges, matte "
           "flat color, diorama depth, hand-made charm.",
           ["papercraft", "cut paper", "layered", "matte"],
           ["smooth", "3d render"], "cut-paper layers", "Craft & Physical"),
        mk("lego_art", "LEGO Art", "Brick-built render look",
           "Style focus: LEGO-built scene. Studded plastic bricks, limited brick palette, "
           "blocky forms, playful miniature diorama lighting.",
           ["lego", "brick", "studded", "blocky"],
           ["organic", "realistic"], "brick diorama", "Craft & Physical"),
        mk("voxel_art", "Voxel Art", "Cube-based 3D look",
           "Style focus: voxel art. Blocky cube-based geometry, limited palette per material, "
           "Minecraft-like readable forms, soft even light.",
           ["voxel art", "cubes", "blocky", "limited palette"],
           ["smooth", "organic"], "blocky cubes", "Craft & Physical"),
        mk("felt_art", "Felt Craft", "Fuzzy felt look",
           "Style focus: felt craft. Soft fuzzy fiber texture, stitched edges, matte pastel "
           "palette, hand-made toy warmth.",
           ["felt", "fuzzy", "stitched", "matte"],
           ["rigid", "glossy"], "fuzzy hand-made", "Craft & Physical"),
        mk("wireframe", "Wireframe", "3D wireframe render",
           "Style focus: 3D wireframe. Visible polygonal edges, glowing lines on dark, structural "
           "blueprint aesthetic, technical beauty.",
           ["wireframe", "polygonal edges", "glowing lines", "blueprint"],
           ["solid", "textured"], "glowing edges", "Craft & Physical"),
        mk("origami", "Origami", "Paper-fold art",
           "Style focus: origami. Crisp folded paper geometry, sharp creases, matte single-sheet "
           "palette, elegant minimal form.",
           ["origami", "folded paper", "sharp crease", "minimal"],
           ["messy", "rough"], "crisp folds", "Craft & Physical"),
    ],
    "decorative": [
        mk("william_morris", "William Morris", "Arts-and-Crafts pattern",
           "Style focus: William Morris pattern. Dense symmetrical foliage, flowering vines, "
           "muted natural palette, intricate repeating wallpaper design.",
           ["william morris", "foliage pattern", "symmetrical", "intricate"],
           ["minimal", "modern"], "dense foliage", "Decorative & Pattern"),
        mk("persian_miniature_pat", "Persian Pattern", "Islamic miniature pattern",
           "Style focus: Persian/Islamic pattern. Interlacing arabesque, geometric tessellation, "
           "jewel palette, infinitely repeating ornamental logic.",
           ["persian pattern", "arabesque", "tessellation", "geometric"],
           ["random", "figurative"], "interlacing geometry", "Decorative & Pattern"),
        mk("celtic_knot", "Celtic Knot", "Celtic knotwork",
           "Style focus: Celtic knotwork. Unbroken interwoven ribbons, symmetric loops, "
           "stone-and-metal palette, ritual ornament.",
           ["celtic knot", "interwoven", "ribbon", "symmetric"],
           ["asymmetry", "modern"], "interwoven ribbons", "Decorative & Pattern"),
        mk("islamic_geometric", "Islamic Geometric", "Tessellated geometry",
           "Style focus: Islamic geometric tilework. Precise star-and-polygon tessellation, "
           "mathematical symmetry, glazed tile palette.",
           ["islamic geometric", "tessellation", "star polygon", "symmetry"],
           ["organic", "random"], "mathematical symmetry", "Decorative & Pattern"),
        mk("batik", "Batik", "Wax-resist textile",
           "Style focus: batik. Wax-resist dye cracks, flowing organic motifs, layered earthy "
           "palette, textile repetition.",
           ["batik", "wax resist", "organic motif", "earthy"],
           ["rigid", "neon"], "crackle motifs", "Decorative & Pattern"),
        mk("folk_art", "Folk Art", "Naive decorative craft",
           "Style focus: folk art. Naive charming forms, bright flat local palette, traditional "
           "motifs, hand-made imperfect delight.",
           ["folk art", "naive", "bright flat", "traditional"],
           ["sophisticated", "photoreal"], "naive charm", "Decorative & Pattern"),
    ],
    "miniature": [
        mk("tilt_shift", "Tilt-Shift", "Miniature fake-diorama photo",
           "Style focus: tilt-shift miniature. Real scene shot with selective blur so it reads as "
           "a tiny model, boosted saturation, toy-like shallow focus band.",
           ["tilt shift", "miniature effect", "selective blur", "boosted saturation"],
           ["deep focus", "realistic scale"], "fake miniature", "Miniature & Toy"),
        mk("toy_photography", "Toy Photography", "Action-figure photo",
           "Style focus: toy photography. Posed action figures, practical mini lighting, forced "
           "perspective, playful cinematic miniature scene.",
           ["toy photography", "action figure", "mini lighting", "forced perspective"],
           ["life size", "flat"], "mini cinematic", "Miniature & Toy"),
        mk("diorama", "Diorama", "Miniature scene model",
           "Style focus: diorama. Tiny detailed modeled environment in a contained scene, careful "
           "scale lighting, collectible-craft polish.",
           ["diorama", "miniature scene", "detailed", "contained"],
           ["vast", "realistic"], "contained detail", "Miniature & Toy"),
        mk("terrarium", "Terrarium", "Glass-dome miniature",
           "Style focus: terrarium. Plants/landscape sealed in glass, soft diffuse light, "
           "miniature ecosystem, calm green enclosure.",
           ["terrarium", "glass dome", "mini ecosystem", "soft light"],
           ["harsh", "vast"], "glass enclosure", "Miniature & Toy"),
        mk("food_miniature", "Food Miniature", "Tiny-food miniature",
           "Style focus: tiny-food miniature. Hyper-detailed miniature cuisine, dollhouse scale, "
           "cute precise crafting, appetizing mini lighting.",
           ["tiny food", "miniature cuisine", "dollhouse", "cute"],
           ["life size", "messy"], "cute tiny food", "Miniature & Toy"),
    ],
}


def main():
    os.makedirs(STYLES_DIR, exist_ok=True)

    # 1) system_prompts.json
    with open(os.path.join(STYLES_DIR, "system_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(SYSTEM_PROMPTS, f, indent=2, ensure_ascii=False)
    print("wrote system_prompts.json")

    # 2) per-direction files
    # Start from migrated presets
    buckets = {}
    for p in _src.get("presets", []):
        fid = p.get("id")
        if fid not in FILE_FOR:
            print("WARN unmapped preset:", fid)
            continue
        fname, group = FILE_FOR[fid]
        sp_neg = p.get("system_prompt_no_negative", "")
        if not sp_neg:
            sp_neg = GEN_NEG
        rec = {
            "id": fid,
            "name": p.get("name", fid),
            "description": p.get("description", ""),
            "system_prompt": p.get("system_prompt", ""),
            "system_prompt_no_negative": sp_neg,
            "style_tags_positive": p.get("style_tags_positive", []),
            "style_tags_negative": p.get("style_tags_negative", []),
            "blend_note": p.get("blend_note", "") or p.get("name", ""),
            "category": group,
            "disabled_in_no_negative_mode": bool(p.get("disabled_in_no_negative_mode", False)),
        }
        buckets.setdefault(fname, []).append(rec)

    # 3) append curated new presets
    for fname, presets in NEW.items():
        for pr in presets:
            if not pr.get("system_prompt_no_negative"):
                pr["system_prompt_no_negative"] = GEN_NEG
            buckets.setdefault(fname, []).append(pr)

    # 4) write each file
    for fname, presets in sorted(buckets.items()):
        presets.sort(key=lambda r: r["name"].lower())
        ids = [r["id"] for r in presets]
        assert len(ids) == len(set(ids)), f"duplicate ids in {fname}: {ids}"
        out = {
            "schema_version": "2.0.0",
            "category": presets[0]["category"] if presets else "",
            "presets": presets,
        }
        with open(os.path.join(STYLES_DIR, fname + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"wrote {fname}.json ({len(presets)} presets)")

    total = sum(len(v) for v in buckets.values())
    print(f"TOTAL presets: {total}")


if __name__ == "__main__":
    main()
