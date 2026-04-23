// ── Hide unused feedback buttons ─────────────────────────────────────
// Chainlit shows thumbs-up / thumbs-down when a data layer is configured
// (needed for thread persistence), but MedMCP does not use feedback.
(function () {
  const SELECTOR =
    ".positive-feedback-on, .positive-feedback-off, .negative-feedback-on, .negative-feedback-off";

  function hide(root) {
    for (const el of root.querySelectorAll(SELECTOR)) {
      el.style.display = "none";
    }
  }

  hide(document);

  new MutationObserver(function (mutations) {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType === 1) {
          if (node.matches && node.matches(SELECTOR)) {
            node.style.display = "none";
          }
          hide(node);
        }
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
})();

// ── Context usage indicator ───────────────────────────────────────────
// Injects a small bar + token label into Chainlit's .watermark footer and
// keeps it up to date via Socket.IO push (ctx_update) with a 250 ms poll fallback.
(function () {
  "use strict";

  // Append the badge to .watermark.  Returns true when successful.
  // Note: querySelectorAll does not test the root element itself, so we use
  // document.querySelector which always searches the full document.
  function tryInject() {
    if (document.getElementById("ctx-indicator")) return true;
    const wm = document.querySelector(".watermark");
    if (!wm) return false;
    const badge = document.createElement("div");
    badge.id = "ctx-indicator";
    badge.innerHTML =
      '<div id="ctx-bar-track"><div id="ctx-bar-fill"></div></div>' +
      '<span id="ctx-label">— / —</span>';
    wm.appendChild(badge);
    return true;
  }

  // Poll every 100 ms until .watermark exists, then stop.
  const injectTimer = setInterval(function () {
    if (tryInject()) clearInterval(injectTimer);
  }, 100);

  // Re-inject if React removes our badge (e.g. on component re-mount).
  new MutationObserver(function () {
    if (!document.getElementById("ctx-indicator")) tryInject();
  }).observe(document.body, { childList: true, subtree: true });

  function updateBadge(used, size) {
    const fill = document.getElementById("ctx-bar-fill");
    const label = document.getElementById("ctx-label");
    if (!fill || !label) return;

    const pct = size > 0 ? Math.min(100, (used / size) * 100) : 0;
    fill.style.width = pct + "%";
    // Shift fill colour from muted → amber → rose as context fills.
    fill.className = pct >= 80 ? "ctx-high" : pct >= 50 ? "ctx-mid" : "";

    const sizeK = Math.round(size / 1000);
    if (used > 0) {
      const usedK = (used / 1000).toFixed(used < 10000 ? 1 : 0);
      label.textContent = usedK + "k / " + sizeK + "k";
    } else {
      label.textContent = "— / " + sizeK + "k";
    }
  }

  // Expose for the socket push handler in the IIFE below.
  window._ctxUpdateBadge = updateBadge;

  async function poll() {
    try {
      const resp = await fetch("/api/context-usage");
      if (!resp.ok) return;
      const data = await resp.json();
      if (typeof data.used === "number" && typeof data.size === "number") {
        updateBadge(data.used, data.size);
      }
    } catch {
      // Ignore — server may still be starting.
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      poll();
      setInterval(poll, 250);
    });
  } else {
    poll();
    setInterval(poll, 250);
  }
})();

// ── Override the ChatSettings Reset button ───────────────────────────
// Restores widget defaults (the `initial` values) rather than the values
// captured when the panel was opened.
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

  // Parse a socket.io frame for events we care about.
  function parseSioFrame(text) {
    if (typeof text !== "string") return;
    // socket.io v4 frames: 42["event_name", payload]
    const settingsMatch = text.match(/42\["chat_settings",(.+?)\](?:\d|$)/s);
    if (settingsMatch) {
      try {
        _defaults = extractDefaults(JSON.parse(settingsMatch[1]));
      } catch {
        // Ignore parse errors on non-matching frames.
      }
    }
    // ctx_update push — real token count from vibe-acp, bypasses the poll.
    const ctxMatch = text.match(/42\["ctx_update",(\{[^}]+\})\]/);
    if (ctxMatch && window._ctxUpdateBadge) {
      try {
        const d = JSON.parse(ctxMatch[1]);
        if (typeof d.used === "number" && typeof d.size === "number") {
          window._ctxUpdateBadge(d.used, d.size);
        }
      } catch {
        // Ignore malformed frames.
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

// ── Make the extensions info textarea read-only ───────────────────────
// The TextInput widget has no readonly prop, so we set the attribute via JS.
(function () {
  function applyReadonly() {
    const el = document.querySelector(
      'textarea[name="extensions_info"], textarea[id="extensions_info"]'
    );
    if (el && !el.hasAttribute("readonly")) el.setAttribute("readonly", "");
  }
  new MutationObserver(applyReadonly).observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
