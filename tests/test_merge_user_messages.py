*** Begin Patch
*** Add File: tests/test_merge_user_messages.py
+import json
+
+from comfyui_llm_prompt_studio.nodes import writer
+
+
+def test_merge_revision_and_retry_messages():
+    # Simulate a chat_completion side-effect that returns valid JSON only after a retry.
+    seq = [
+        json.dumps({"positive": "a cat", "negative": "b", "scene_name": ""}),
+        json.dumps({"positive": "a cat", "negative": "b", "scene_name": "cat_scene"}),
+    ]
+
+    calls = {
+        "i": 0
+    }
+
+    def side_effect(server_url, api_key, model, messages, temperature, max_tokens, seed=None):
+        # Ensure we never send two consecutive user messages (last two roles must not be user,user)
+        roles = [m.get("role") for m in messages]
+        # If there are at least two messages, the last two should not both be 'user'
+        if len(roles) >= 2:
+            assert not (roles[-1] == "user" and roles[-2] == "user"), "Two consecutive user messages sent"
+        val = seq[calls["i"]]
+        calls["i"] += 1
+        return val
+
+    # Patch writer.ensure_model_loaded out and use our side-effect
+    writer.ensure_model_loaded = lambda *a, **k: None
+    writer.chat_completion = side_effect
+
+    # Run with revision_notes and allow one retry (missing scene_name on first reply)
+    res = writer.LLMPromptStudioWriter().execute(
+        server_url="http://localhost:1234/v1", api_key="", model="m",
+        context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
+        revision_notes="fix this", temperature=0.7, max_tokens=512, seed=0,
+        reuse_last_prompt=False, generate_face_prompts=False, max_field_retries=1,
+        face_prompt_instruction="", prompt_mode="standard", family="", unique_id="u1"
+    )
+
+    assert res[0] == "a cat"
+
*** End Patch
