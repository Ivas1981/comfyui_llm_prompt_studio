@@
 logger = logging.getLogger("llm_prompt_studio")
 
+# Helper: append a user message, or merge into the last user message if one already exists.
+def _append_or_merge_user(messages, text):
+    if messages and messages[-1].get("role") == "user":
+        prev = messages[-1]["content"]
+        # If previous content is a multimodal list, add a text part
+        if isinstance(prev, list):
+            prev.append({"type": "text", "text": text})
+            messages[-1]["content"] = prev
+        else:
+            messages[-1]["content"] = f"{prev}\n\n{text}"
+    else:
+        messages.append({"role": "user", "content": text})
+
@@
-        if revision_notes.strip():
-            messages.append({"role": "user", "content": (
-                "The previous version of the prompt did not pass the critic's check. "
-                f"Requested fixes: {revision_notes}\n"
-                "Generate a CORRECTED prompt taking these fixes into account, "
-                "in the same JSON format."
-            )})
+        if revision_notes.strip():
+            _append_or_merge_user(messages, (
+                "The previous version of the prompt did not pass the critic's check. "
+                f"Requested fixes: {revision_notes}\n"
+                "Generate a CORRECTED prompt taking these fixes into account, "
+                "in the same JSON format."
+            ))
@@
-            messages.append({"role": "user", "content":
-                f"You omitted the required JSON field(s): {', '.join(missing)}. "
-                f"Respond again with a COMPLETE JSON object containing ALL required fields."})
+            _append_or_merge_user(messages,
+                f"You omitted the required JSON field(s): {', '.join(missing)}. "
+                f"Respond again with a COMPLETE JSON object containing ALL required fields.")
