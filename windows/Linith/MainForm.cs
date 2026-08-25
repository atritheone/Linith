using System;
using System.IO;
using System.Drawing;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace Linith
{
  public partial class MainForm : Form
  {
    private readonly WebView2 _web = new WebView2 { Dock = DockStyle.Fill };
    private readonly Panel _splash; // covers white flash before first paint

    private bool _isFullscreen = false;
    private FormWindowState _prevState;
    private FormBorderStyle _prevBorder;
    private bool _prevTopMost;
    private Rectangle _prevBounds;

    // app background (#0f0f12)
    private static readonly Color AppBg = Color.FromArgb(0x0F, 0x0F, 0x12);

    public MainForm()
    {
      InitializeComponent();

      Text = "Linith";
      KeyPreview = true;                                // receive F11 even if webview has focus
      MinimumSize = new Size(800, 600);                 // sensible min
      DoubleBuffered = true;

      // Window/taskbar icon (already copied to output via csproj)
      ShowIcon = true;
      ShowInTaskbar = true;
      var icoPath = Path.Combine(AppContext.BaseDirectory, "Assets", "linith_logo.ico");
      if (File.Exists(icoPath))
        Icon = new Icon(icoPath);

      // Set the form background immediately (prevents white before WebView initializes)
      BackColor = AppBg;

      // Set the WebView control bg (prevents white while initializing)
      _web.BackColor = AppBg;

      // Simple overlay to mask any remaining flash until DOM is ready
      _splash = new Panel
      {
        Dock = DockStyle.Fill,
        BackColor = AppBg,
        Visible = true
      };

      // Order: add WebView first, then splash and bring it to front
      Controls.Add(_web);
      Controls.Add(_splash);
      _splash.BringToFront();

      Load += async (_, __) =>
      {
        // Prefer Fixed WebView2 runtime (./WebView2Fixed), else system runtime
        var baseDir   = AppContext.BaseDirectory;
        var fixedPath = Path.Combine(baseDir, "WebView2Fixed");
        var userData  = Path.Combine(baseDir, "wv2-user");

        CoreWebView2Environment env = await CoreWebView2Environment.CreateAsync(
          browserExecutableFolder: Directory.Exists(fixedPath) ? fixedPath : null,
          userDataFolder: userData,
          options: null);

        await _web.EnsureCoreWebView2Async(env);

        // Ensure WebView’s internal default background is also dark
        // (A,R,G,B: alpha must be 0xFF for full opacity)
        _web.DefaultBackgroundColor = Color.FromArgb(0xFF, AppBg.R, AppBg.G, AppBg.B);

        ConfigureWebView(_web.CoreWebView2);

        // Map a virtual host to Assets so relative paths work consistently
        var assets = Path.Combine(baseDir, "Assets");
        _web.CoreWebView2.SetVirtualHostNameToFolderMapping(
          "app.local", assets, CoreWebView2HostResourceAccessKind.DenyCors);

        // Inject very-early CSS/JS so the document is never white before your CSS loads
        await _web.CoreWebView2.AddScriptToExecuteOnDocumentCreatedAsync(@"
          (function(){
            try {
              document.documentElement.style.backgroundColor = '#0f0f12';
              if (document.body) document.body.style.backgroundColor = '#0f0f12';
            } catch(_) {}
            // capture F11 / Alt+Enter / Esc and notify host for fullscreen toggle
            document.addEventListener('keydown', function(e){
              if (e.code === 'F11' || (e.code === 'Enter' && e.altKey)) {
                e.preventDefault();
                if (window.chrome && window.chrome.webview) {
                  window.chrome.webview.postMessage('toggleFullscreen');
                }
              } else if (e.code === 'Escape') {
                if (window.chrome && window.chrome.webview) {
                  window.chrome.webview.postMessage('escapeFullscreen');
                }
              }
            }, true);
          })();
        ");

        _web.CoreWebView2.WebMessageReceived += (_, msg) =>
        {
          var payload = msg.TryGetWebMessageAsString();
          if (payload == "toggleFullscreen")
            BeginInvoke(new Action(ToggleFullscreen));
          else if (payload == "escapeFullscreen" && _isFullscreen)
            BeginInvoke(new Action(ToggleFullscreen));
        };

        // Hide splash as soon as the DOM is ready (earlier than full navigation)
        _web.CoreWebView2.DOMContentLoaded += (_, __2) =>
          BeginInvoke(new Action(() => _splash.Visible = false));

        // Navigate to your HTML
        _web.CoreWebView2.Navigate("https://app.local/linith_0.232.html");
      };

      KeyDown += MainForm_KeyDown;
    }

    private void ConfigureWebView(CoreWebView2 cwv)
    {
      cwv.Settings.IsZoomControlEnabled = false;   // prevent Ctrl+Wheel changing layout
      cwv.Settings.AreDefaultContextMenusEnabled = true;
      cwv.Settings.AreDevToolsEnabled = true;      // set false for production if desired
    }

    private void MainForm_KeyDown(object? sender, KeyEventArgs e)
    {
      // F11 or Alt+Enter = toggle fullscreen
      if (e.KeyCode == Keys.F11 || (e.KeyCode == Keys.Enter && e.Alt))
      {
        ToggleFullscreen();
        e.Handled = true;
      }
      // ESC exits fullscreen
      else if (e.KeyCode == Keys.Escape && _isFullscreen)
      {
        ToggleFullscreen();
        e.Handled = true;
      }
      // Manual zoom shortcuts (Ctrl+0 / Ctrl+= / Ctrl+-), since Ctrl+Wheel is disabled
      else if (e.Control && e.KeyCode == Keys.D0)
      {
        _web.ZoomFactor = 1.0;
        e.Handled = true;
      }
      else if (e.Control && (e.KeyCode == Keys.Oemplus || e.KeyCode == Keys.Add))
      {
        _web.ZoomFactor = Math.Min(3.0, _web.ZoomFactor + 0.1);
        e.Handled = true;
      }
      else if (e.Control && (e.KeyCode == Keys.OemMinus || e.KeyCode == Keys.Subtract))
      {
        _web.ZoomFactor = Math.Max(0.25, _web.ZoomFactor - 0.1);
        e.Handled = true;
      }
    }

    protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
    {
      // Belt-and-braces: also catch at the message loop
      if (keyData == Keys.F11 || keyData == (Keys.Alt | Keys.Enter))
      {
        ToggleFullscreen();
        return true;
      }
      if (_isFullscreen && keyData == Keys.Escape)
      {
        ToggleFullscreen();
        return true;
      }
      return base.ProcessCmdKey(ref msg, keyData);
    }

    private void ToggleFullscreen()
    {
      if (!_isFullscreen)
      {
        _prevState = WindowState;
        _prevBorder = FormBorderStyle;
        _prevTopMost = TopMost;
        _prevBounds = Bounds;

        FormBorderStyle = FormBorderStyle.None;
        TopMost = false; // set true for kiosk
        WindowState = FormWindowState.Normal;
        Bounds = Screen.FromControl(this).Bounds; // fill current monitor
        _isFullscreen = true;
      }
      else
      {
        FormBorderStyle = _prevBorder;
        TopMost = _prevTopMost;
        WindowState = _prevState;
        Bounds = _prevBounds;
        _isFullscreen = false;
      }
    }
  }
}
