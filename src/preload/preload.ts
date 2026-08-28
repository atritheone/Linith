import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("linithDesktop", {
  appName: "Linith",
  platform: process.platform
});
