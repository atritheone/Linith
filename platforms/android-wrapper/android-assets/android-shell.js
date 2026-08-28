document.body.classList.add("linith-android");

document.addEventListener("touchmove", (event) => {
  if (event.scale && event.scale !== 1) event.preventDefault();
}, { passive: false });
