// Hide the unused thumbs-up / thumbs-down feedback buttons on assistant messages.
// Chainlit shows these when a data layer is configured (needed for thread persistence),
// but MedMCP does not use the feedback feature.
(function () {
  const SELECTOR =
    ".positive-feedback-on, .positive-feedback-off, .negative-feedback-on, .negative-feedback-off";

  function hide(root) {
    for (const el of root.querySelectorAll(SELECTOR)) {
      el.style.display = "none";
    }
  }

  // Catch elements already in the DOM.
  hide(document);

  // Catch elements added later by React renders.
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
