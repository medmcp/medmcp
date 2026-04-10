// Override the Reset button in the ChatSettings panel so that it restores
// widget defaults (the `initial` values) rather than the values captured when
// the panel was opened.
(function () {
  "use strict";

  // Default values extracted from widget `initial` fields, keyed by widget id.
  let _defaults = {};

  // ── Capture widget definitions from the socket ──────────────────────

  function extractDefaults(inputs) {
    const defaults = {};
    for (const input of inputs) {
      if (Array.isArray(input?.inputs)) {
        // Tab — recurse into its children.
        Object.assign(defaults, extractDefaults(input.inputs));
      } else if (input?.id !== undefined) {
        defaults[input.id] = input.initial;
      }
    }
    return defaults;
  }

  // Parse a socket.io frame for a `chat_settings` event and store the defaults.
  function parseSioFrame(text) {
    if (typeof text !== "string") return;
    // socket.io v4 frames: 42["event_name", payload]
    const match = text.match(/42\["chat_settings",(.+?)\](?:\d|$)/s);
    if (match) {
      try {
        _defaults = extractDefaults(JSON.parse(match[1]));
      } catch {
        // Ignore parse errors on non-matching frames.
      }
    }
  }

  // Hook transports to intercept socket.io frames carrying `chat_settings`.
  // socket.io starts with HTTP long-polling and upgrades to WebSocket, so we
  // need to intercept both.
  function hookTransports() {
    // ── WebSocket ──
    const OrigWS = window.WebSocket;
    const hookedSockets = new WeakSet();

    window.WebSocket = function (...args) {
      const ws = new OrigWS(...args);
      if (!hookedSockets.has(ws)) {
        hookedSockets.add(ws);
        ws.addEventListener("message", function (evt) {
          parseSioFrame(evt.data);
        });
      }
      return ws;
    };
    window.WebSocket.prototype = OrigWS.prototype;
    Object.assign(window.WebSocket, OrigWS);

    // ── XMLHttpRequest (long-polling fallback) ──
    const origOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (...args) {
      // Only hook requests that look like socket.io polling.
      const url = args[1];
      if (typeof url === "string" && url.includes("socket.io")) {
        this.addEventListener("load", function () {
          parseSioFrame(this.responseText);
        });
      }
      return origOpen.apply(this, args);
    };
  }

  // ── Override the Reset button ───────────────────────────────────────

  function overrideResetButton() {
    const observer = new MutationObserver(() => {
      // The settings panel is rendered inside a dialog with id="chat-settings"
      // (modal mode) or a sidebar section with id="chat-settings-sidebar-content".
      const panels = document.querySelectorAll(
        "#chat-settings, #chat-settings-sidebar-content"
      );
      for (const panel of panels) {
        const buttons = panel.querySelectorAll("button");
        for (const btn of buttons) {
          if (btn.dataset.resetPatched) continue;
          // Skip Confirm and Cancel buttons.
          if (btn.id === "confirm" || btn.id === "confirm-sidebar") continue;
          // The Reset button uses variant="outline" (has border classes, no ghost).
          if (!btn.className.includes("border") || btn.className.includes("ghost"))
            continue;
          // Skip switch buttons (they also live inside the panel).
          if (btn.getAttribute("role") === "switch") continue;

          btn.dataset.resetPatched = "true";
          btn.addEventListener(
            "click",
            function (e) {
              e.stopImmediatePropagation();
              e.preventDefault();
              resetToDefaults(panel);
            },
            true
          );
        }
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  function resetToDefaults(panel) {
    if (Object.keys(_defaults).length === 0) return;

    for (const [id, defaultValue] of Object.entries(_defaults)) {
      // Switch widgets — Radix UI renders <button role="switch"> with
      // data-state="checked" / "unchecked".
      const switchBtn = panel.querySelector(`button[role="switch"][id="${id}"]`);
      if (switchBtn) {
        const isChecked =
          switchBtn.getAttribute("data-state") === "checked" ||
          switchBtn.getAttribute("aria-checked") === "true";
        const shouldBeChecked = !!defaultValue;
        if (isChecked !== shouldBeChecked) {
          switchBtn.click();
        }
        continue;
      }

      // Text / number inputs — set value via the native setter so React picks
      // up the change.
      const input = panel.querySelector(`input[name="${id}"], input[id="${id}"]`);
      if (input) {
        const proto =
          input.tagName === "TEXTAREA"
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
        if (setter) {
          setter.call(input, defaultValue ?? "");
          input.dispatchEvent(new Event("input", { bubbles: true }));
          input.dispatchEvent(new Event("change", { bubbles: true }));
        }
      }
    }
  }

  // ── Bootstrap ───────────────────────────────────────────────────────

  hookTransports();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", overrideResetButton);
  } else {
    overrideResetButton();
  }
})();
