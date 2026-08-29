import { initGame } from "./game/game";
import { initDynamicLayout } from "./ui/layout";
import { initSettings } from "./ui/settings";

const version = document.getElementById("version");
if (version) {
  version.textContent = __APP_VERSION__.replace(/\.0$/, "");
}

initialise("settings", initSettings);
initialise("layout", initDynamicLayout);
initialise("game", initGame);

function initialise(name: string, init: () => void): void {
  try {
    init();
  } catch (error) {
    console.error(`[Linith] Failed to initialise ${name}.`, error);
  }
}
