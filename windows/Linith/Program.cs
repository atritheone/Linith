using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Linith
{
  internal static class Program
  {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern int SetCurrentProcessExplicitAppUserModelID(string appID);

    [STAThread]
    static void Main()
    {
      // Taskbar grouping/pin identity (no username)
      SetCurrentProcessExplicitAppUserModelID("com.linith.desktop");

      // Optional: quiet DPI manifest warning here instead of manifest
      Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);

      ApplicationConfiguration.Initialize();
      Application.Run(new MainForm());
    }
  }
}
