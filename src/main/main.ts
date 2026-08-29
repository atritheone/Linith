import { app, BrowserWindow } from "electron";
import { join } from "node:path";

let mainWindow: BrowserWindow | null = null;

function getAppIconPath(): string {
  const iconFile = process.platform === "win32" ? "icon.ico" : "icon.png";
  return app.isPackaged
    ? join(process.resourcesPath, iconFile)
    : join(__dirname, "../../build", iconFile);
}

function createMainWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: "#080a0d",
    fullscreenable: true,
    autoHideMenuBar: true,
    show: false,
    useContentSize: true,
    icon: getAppIconPath(),
    webPreferences: {
      preload: join(__dirname, "../preload/preload.js"),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.maximize();
    mainWindow?.show();
  });

  mainWindow.webContents.on("before-input-event", (event, input) => {
    if (!mainWindow) {
      return;
    }

    const key = input.key.toLowerCase();
    const toggleFullscreen =
      (key === "f11" && !input.control && !input.meta && !input.shift && !input.alt) ||
      (key === "enter" && input.alt && !input.control && !input.meta && !input.shift);

    if (toggleFullscreen) {
      event.preventDefault();
      mainWindow.setFullScreen(!mainWindow.isFullScreen());
    } else if (key === "escape" && mainWindow.isFullScreen()) {
      event.preventDefault();
      mainWindow.setFullScreen(false);
    }
  });

  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (event) => {
    event.preventDefault();
  });
  mainWindow.webContents.session.setPermissionCheckHandler(() => false);
  mainWindow.webContents.session.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });

  const rendererUrl = process.env.ELECTRON_RENDERER_URL;
  if (rendererUrl) {
    void mainWindow.loadURL(rendererUrl);
  } else {
    void mainWindow.loadFile(join(__dirname, "../renderer/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
