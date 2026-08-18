"""aiohttp routes exposed to the web JS bridge (/llm_prompt_studio/*)."""
from aiohttp import web
from server import PromptServer

from .library import load_library, resolve_library_path, save_prompt_to_library
from .lm_http import cache_models, fetch_models, server_status


@PromptServer.instance.routes.post("/llm_prompt_studio/models")
async def llm_prompt_studio_models(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    server_url = data.get("server_url", "http://localhost:1234/v1")
    api_key = data.get("api_key", "")
    try:
        models = fetch_models(server_url, api_key)
        cache_models(server_url, api_key, models)
        return web.json_response({"models": models})
    except Exception as e:
        return web.json_response({"models": [], "error": str(e)})


@PromptServer.instance.routes.get("/llm_prompt_studio/status")
async def llm_prompt_studio_status(request):
    """Report LM Studio reachability and what is loaded, for the front-end indicator.

    Reads ``server_url`` / ``api_key`` from the query string (or falls back to the
    default local server). ``server_status`` never raises, so this route always
    returns a JSON object the JS widget can render directly."""
    server_url = request.query.get("server_url", "http://localhost:1234/v1")
    api_key = request.query.get("api_key", "")
    try:
        status = server_status(server_url, api_key)
    except Exception as e:  # noqa: BLE001 - never let the route crash
        status = {"reachable": False, "loaded_models": [], "error": str(e)}
    return web.json_response(status)


@PromptServer.instance.routes.post("/llm_prompt_studio/library/save")
async def llm_prompt_studio_library_save(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    library_path = data.get("library_path", "")
    scene_name = data.get("scene_name", "")
    positive = data.get("positive", "")
    negative = data.get("negative", "")
    face_positive = data.get("face_positive", "")
    face_negative = data.get("face_negative", "")
    try:
        lib = resolve_library_path(library_path)
        name, added = save_prompt_to_library(lib, scene_name, positive, negative,
                                             face_positive, face_negative)
        return web.json_response({"name": name, "added": added})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/llm_prompt_studio/presets")
async def llm_prompt_studio_presets(request):
    try:
        from .presets import load_presets, get_user_presets_path, get_preset_names
        load_presets()  # ensure migrated user file exists
        return web.json_response({
            "names": get_preset_names(),
            "path": get_user_presets_path(),
        })
    except Exception as e:
        return web.json_response({"names": [], "error": str(e)})


@PromptServer.instance.routes.post("/llm_prompt_studio/presets/reset")
async def llm_prompt_studio_presets_reset(request):
    try:
        from .presets import reset_to_defaults
        reset_to_defaults()
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


@PromptServer.instance.routes.get("/llm_prompt_studio/library/scenes")
async def llm_prompt_studio_library_scenes(request):
    library_path = request.query.get("library_path", "")
    try:
        lib = resolve_library_path(library_path)
        entries = load_library(lib)
        names = [str(e.get("name")) for e in entries
                 if isinstance(e, dict) and e.get("name")]
        return web.json_response({"scenes": names})
    except Exception as e:
        return web.json_response({"scenes": [], "error": str(e)})


@PromptServer.instance.routes.get("/llm_prompt_studio/sampler_params")
async def llm_prompt_studio_sampler_params(request):
    """Recommend sampler parameters for a checkpoint family / preset.

    Query params: ``family``, ``preset`` (balanced|speed|quality|user),
    ``ckpt`` (filename), ``arch`` (base architecture). The studio KSampler supports
    the full scheduler list (standard + AYS SD1/SDXL/SVD + GITS), so there is no
    ``target`` parameter - the returned scheduler always matches the destination.
    """
    family = request.query.get("family", "")
    preset = request.query.get("preset", "balanced")
    ckpt = request.query.get("ckpt", "")
    arch = request.query.get("arch", "")
    try:
        from .nodes._distilled_presets import recommend
        rec = recommend(family, preset, ckpt or "", architecture=arch or "")
        return web.json_response(rec)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/llm_prompt_studio/library/scene")
async def llm_prompt_studio_library_scene(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    library_path = data.get("library_path", "")
    scene_name = data.get("scene_name", "")
    try:
        lib = resolve_library_path(library_path)
        entries = load_library(lib)
        for e in entries:
            if isinstance(e, dict) and e.get("name") == scene_name:
                return web.json_response(e)
        return web.json_response({"error": "Scene not found"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
