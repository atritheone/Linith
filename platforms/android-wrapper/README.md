# Linith Android Wrapper

This Capacitor 8 project consumes the shared renderer build from `../../dist/renderer`. Game source belongs in the repository-root `src` tree; `www` is generated and must not be edited directly. Native builds require JDK 21 and Android SDK 36.

From the repository root, build the renderer:

```powershell
npm ci
npm test
npm run build
```

Then update Android's generated web assets:

```powershell
cd platforms/android-wrapper
npm ci
npm run sync
npm run open
```

`npm run copy:web` copies the Vite renderer and layers the portrait-only Android shell from `android-assets` onto it. `npm run apk:debug` also builds a debug APK on Windows.

Gradle output is kept in `.android-build/` so OneDrive does not contend with Android's conventional nested `app/build` tree. The debug APK is written to `.android-build/app/outputs/apk/debug/app-debug.apk`.
