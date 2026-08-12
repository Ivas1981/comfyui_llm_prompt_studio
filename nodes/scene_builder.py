@@
 logger = logging.getLogger("llm_prompt_studio")
 
+# Local helper to append or merge user messages (keeps role sequence valid for strict servers)
+def _append_or_merge_user(messages, text):
+    if messages and messages[-1].get("role") == "user":
+        prev = messages[-1]["content"]
+        if isinstance(prev, list):
+            prev.append({"type": "text", "text": text})
+            messages[-1]["content"] = prev
+        else:
+            messages[-1]["content"] = f"{prev}\n\n{text}"
+    else:
+        messages.append({"role": "user", "content": text})
@@
-            messages.append({"role": "user", "content":
-                f"You omitted the required JSON field(s): {', '.join(missing)}. "
-                f"Respond again with a COMPLETE JSON object containing ALL required fields."})
+            _append_or_merge_user(messages,
+                f"You omitted the required JSON field(s): {', '.join(missing)}. "
+                f"Respond again with a COMPLETE JSON object containing ALL required fields.")
