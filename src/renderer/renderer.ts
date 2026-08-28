import { initGame } from "./game/game";
import { initDynamicLayout } from "./ui/layout";
import { initSettings } from "./ui/settings";

const version = document.getElementById("version");
if (version) {
  version.textContent = __APP_VERSION__;
}

initGame();
initSettings();
initDynamicLayout();
