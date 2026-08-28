const FOOTER_GAP = 6;
const HINT_GAP = 8;

export function initDynamicLayout(): void {
  let animationFrame = 0;
  let timeout = 0;

  const adjust = (): void => {
    window.cancelAnimationFrame(animationFrame);
    window.clearTimeout(timeout);
    animationFrame = window.requestAnimationFrame(() => {
      timeout = window.setTimeout(() => {
        adjustFooterPosition();
        adjustHintMaxHeight();
      });
    });
  };

  window.addEventListener("load", adjust, { passive: true });
  window.addEventListener("resize", adjust, { passive: true });

  const mutationObserver = new MutationObserver(adjust);
  mutationObserver.observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["class", "hidden"]
  });

  const resizeObserver = new ResizeObserver(adjust);
  for (const element of document.querySelectorAll<HTMLElement>(".frame, .panel, #board, .hint, .footer")) {
    resizeObserver.observe(element);
  }

  window.setTimeout(adjust, 50);
}

function adjustFooterPosition(): void {
  const footer = document.querySelector<HTMLElement>(".footer");
  const board = document.querySelector<HTMLElement>("#board");
  if (!footer || !board) {
    return;
  }

  const boardBounds = board.getBoundingClientRect();
  const footerHeight = Math.max(footer.getBoundingClientRect().height, 0);
  const desiredBottom = Math.floor(window.innerHeight - boardBounds.bottom - FOOTER_GAP - footerHeight);
  footer.style.bottom = `${Math.max(0, Number.isFinite(desiredBottom) ? desiredBottom : 0)}px`;
}

function adjustHintMaxHeight(): void {
  const hint = document.querySelector<HTMLElement>(".hint");
  const footer = document.querySelector<HTMLElement>(".footer");
  if (!hint || !footer) {
    return;
  }

  const hintTop = hint.getBoundingClientRect().top;
  const footerTop = footer.getBoundingClientRect().top;
  const available = Math.max(80, Math.floor(footerTop - hintTop - HINT_GAP));
  const safetyCap = Math.max(120, Math.floor(footerTop - 12));
  hint.style.maxHeight = `${Math.min(available, safetyCap)}px`;
  hint.style.overflowY = "auto";
}
