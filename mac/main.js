const { app, BrowserWindow } = require('electron');
const path = require('path');

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#0f0f12',
    fullscreenable: true,
    autoHideMenuBar: true,
    show: false,
    useContentSize: true,
    webPreferences: {
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile(path.join(__dirname, 'static', 'linith_0.232.html'));

  // Maximize window once ready
  win.once('ready-to-show', () => {
    win.maximize();
    win.show();
  });

  // Handle fullscreen toggles
  win.webContents.on('before-input-event', (event, input) => {
    const key = input.key.toLowerCase();

    const isF11 =
      key === 'f11' &&
      !input.control && !input.meta && !input.shift && !input.alt;

    const isAltEnter =
      key === 'enter' &&
      input.alt &&
      !input.control && !input.meta && !input.shift;

    const isEscape =
      key === 'escape' &&
      !input.control && !input.meta && !input.shift && !input.alt;

    if (isF11 || isAltEnter) {
      event.preventDefault();
      win.setFullScreen(!win.isFullScreen());
    }

    if (isEscape && win.isFullScreen()) {
      event.preventDefault();
      win.setFullScreen(false);
    }
  });

  win.on('closed', () => {
    win = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // macOS keeps the app alive unless Cmd+Q is pressed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (win === null) {
    createWindow();
  }
});
