document.body.classList.add("linith-android");

const boardWrap = document.querySelector(".boardWrap");
const startMenu = document.getElementById("startMenu");
if (boardWrap && startMenu) {
  boardWrap.appendChild(startMenu);
}

document.addEventListener("touchmove", (event) => {
  if (event.scale && event.scale !== 1) event.preventDefault();
}, { passive: false });
