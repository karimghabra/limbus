"""The LIMBUS installer, end to end: install, use, upgrade, uninstall.

    python e2e/run_e2e.py --installer dist/LIMBUS-Setup-0.1.0.exe [--work DIR]
                          [--dir DIR] [--documents] [--desktop] [--keep-installed]

Runs with any Python 3.8+ on Windows (standard library only), and treats the
machine as the naive laptop the installer is for: everything the installed
app runs gets an environment with no Python and no ffmpeg on PATH and no
PYTHON* variables, so it can only work with what the installer put down.

  1. check a folder too deep for Windows' path limit is refused up front;
     then install silently, for the current user, where the installer
     installs by default (or into --dir: CI uses one with a space in it, to
     check the shortcuts quote it); check the files, the Start-menu
     shortcuts and the entry in Apps & features
  2. install again over it, as an upgrade does
  3. run e2e/app_scenarios.py with the installed Python: the app itself,
     recording, analysis and the rest (see that file)
  4. start the app the way its Start-menu shortcut does (pythonw, no
     console): its window must open, stay up and close cleanly, with nothing
     but its startup line in the log
  5. uninstall silently: the folder, shortcuts and Apps & features entry go,
     the recordings stay

--documents lets step 4 use the real Documents\\LIMBUS\\recordings (CI does;
by default it records under the work folder instead). --desktop also asks for
the desktop shortcut. Exit code 0 when every check passes; report.json in the
work folder lists them.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import winreg

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ID = "{12D5E0C6-4707-412F-9370-4E23C3BFA622}"   # installer/limbus.iss
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_ID}_is1"
START_MENU = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs")
DESKTOP = os.path.join(os.environ["USERPROFILE"], "Desktop")
LOG_PATH = os.path.join(os.environ["LOCALAPPDATA"], "LIMBUS", "limbus.log")   # camera_recorder.LOG_PATH
NAIVE_KEEP = {k.upper() for k in (
    "SystemRoot", "SystemDrive", "windir", "TEMP", "TMP", "USERPROFILE", "USERNAME",
    "USERDOMAIN", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH", "ProgramData",
    "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles",
    "CommonProgramFiles(x86)", "CommonProgramW6432", "ALLUSERSPROFILE", "PUBLIC",
    "COMPUTERNAME", "NUMBER_OF_PROCESSORS", "OS", "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "PATHEXT", "ComSpec",
    "SESSIONNAME", "LOGONSERVER")}

results = []


def check(ok, what, detail=""):
    results.append({"check": what, "status": "ok" if ok else "FAIL", "detail": detail})
    print(f"  {'ok' if ok else 'FAIL':5s} {what}" + (f"  [{detail}]" if detail and not ok else ""),
          flush=True)
    return ok


def section(title):
    print(f"\n[{time.strftime('%H:%M:%S')}] {title}", flush=True)


def guarded(fn, *args):
    """Run one step; an exception in it is a failed check, not the end of
    the run (the uninstall and the report still happen)."""
    try:
        return fn(*args)
    except Exception as exc:
        traceback.print_exc()
        check(False, f"{fn.__name__} raised {type(exc).__name__}: {exc}")
        return None


def naive_env(**extra):
    """This machine's environment as a laptop with nothing installed would
    have it: Windows' own folders on PATH, no Python or ffmpeg anywhere."""
    env = {k: v for k, v in os.environ.items() if k.upper() in NAIVE_KEEP}
    root = env.get("SYSTEMROOT") or env.get("SystemRoot") or r"C:\Windows"
    env["PATH"] = os.pathsep.join([os.path.join(root, "System32"), root,
                                   os.path.join(root, "System32", "Wbem"),
                                   os.path.join(root, "System32", "WindowsPowerShell", "v1.0")])
    env.update(extra)
    return env


def shortcut(path):
    """(target, arguments, working folder, icon) of a .lnk file."""
    ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); "
          "@($s.TargetPath, $s.Arguments, $s.WorkingDirectory, $s.IconLocation) -join \"`n\"")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         env=dict(os.environ, LNK=path), capture_output=True, text=True)
    return (out.stdout.rstrip("\r\n").split("\n") + ["", "", "", ""])[:4]


def uninstall_entry():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as k:
            return {name: winreg.QueryValueEx(k, name)[0]
                    for name in ("DisplayName", "DisplayVersion", "InstallLocation", "UninstallString")}
    except OSError:
        return None


def run_installer(installer, dest, log, desktop):
    """Install silently for the current user; into `dest`, or where the
    installer puts it by default when that is None."""
    t0 = time.time()
    rc = subprocess.run([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER",
                         *([f"/DIR={dest}"] if dest else []), f"/LOG={log}",
                         "/TASKS=" + ("desktopicon" if desktop else "")]).returncode
    return rc, time.time() - t0


def refuses_deep_folder(installer, work):
    """A folder so deep that the app's files would pass Windows' path limit
    is refused before anything is copied, with a reason in the log."""
    deep = os.path.join(work, "deep")
    deep = os.path.join(deep, "d" * (230 - len(deep)), "LIMBUS")
    log = os.path.join(work, "deep.log")
    rc, _ = run_installer(installer, deep, log, False)
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            said = "too deep" in f.read()
    except OSError:
        said = False
    check(rc != 0 and not os.path.exists(deep) and uninstall_entry() is None and said,
          f"a folder too deep for Windows' path limit is refused up front (exit code {rc})",
          f"created: {os.path.exists(deep)}; reason logged: {said}")


# ---- windows of a process (for the shortcut launch) ------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.OpenProcess.restype = wt.HANDLE
EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
WM_CLOSE = 0x0010
SYNCHRONIZE, QUERY_LIMITED = 0x00100000, 0x1000
WAIT_TIMEOUT = 0x102


def windows_titled(title):
    """[(hwnd, pid)] of the visible top-level windows with this title."""
    found = []

    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value == title:
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                found.append((hwnd, pid.value))
        return True

    user32.EnumWindows(EnumWindowsProc(visit), 0)
    return found


def process_image(handle):
    buf = ctypes.create_unicode_buffer(1024)
    n = wt.DWORD(len(buf))
    kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(n))
    return buf.value


def wait_process(handle, seconds):
    """The exit code, or None if still running after `seconds`."""
    if kernel32.WaitForSingleObject(handle, int(seconds * 1000)) == WAIT_TIMEOUT:
        return None
    code = wt.DWORD()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    return code.value


# ---- the steps ----------------------------------------------------------------------

def check_install(dest, version, desktop):
    runtime, app = os.path.join(dest, "runtime"), os.path.join(dest, "app")
    for rel in (r"runtime\python.exe", r"runtime\pythonw.exe", r"runtime\msvcp140.dll",
                r"app\camera_recorder.py", r"app\assets\limbus.ico", r"app\e2e\app_scenarios.py",
                r"app\analysis\stabilize\__main__.py", r"app\vesselmap\__init__.py", "unins000.exe"):
        check(os.path.isfile(os.path.join(dest, rel)), f"installed {rel}")
    try:
        with open(os.path.join(app, "build_info.json"), encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError) as exc:
        check(False, "build_info.json readable", str(exc))
        info = {}
    check(info.get("version") == version, f"build_info.json says version {info.get('version')}", version)
    entry = uninstall_entry()
    check(entry is not None and entry.get("DisplayVersion") == version,
          f"listed in Apps & features: {entry and entry.get('DisplayName')}", json.dumps(entry))
    want_args = f'-s "{os.path.join(app, "camera_recorder.py")}"'
    for name, exe in (("LIMBUS", "pythonw.exe"), ("LIMBUS Self-Test", "python.exe")):
        lnk = os.path.join(START_MENU, f"{name}.lnk")
        if not check(os.path.isfile(lnk), f"Start-menu shortcut {name}", lnk):
            continue
        target, args, workdir, icon = shortcut(lnk)
        check(os.path.normcase(target) == os.path.normcase(os.path.join(runtime, exe)),
              f"{name} runs the installed {exe}", target)
        if name == "LIMBUS":
            check(args == want_args, f"{name} starts camera_recorder.py ({args})", want_args)
        check(os.path.normcase(workdir) == os.path.normcase(app) and icon.lower().split(",")[0].endswith(
            "limbus.ico"), f"{name} has its working folder and icon", f"{workdir} {icon}")
    if desktop:
        check(os.path.isfile(os.path.join(DESKTOP, "LIMBUS.lnk")), "desktop shortcut")
    return info


def run_scenarios(dest, work):
    report = os.path.join(work, "scenarios.json")
    env = naive_env(PYTHONIOENCODING="utf-8")
    python = os.path.join(dest, "runtime", "python.exe")
    check(shutil.which("python", path=env["PATH"]) is None and shutil.which("ffmpeg", path=env["PATH"]) is None,
          "the test environment has no Python and no ffmpeg on PATH")
    cmd = [python, "-s", os.path.join(dest, "app", "e2e", "app_scenarios.py"), "--expect-installed",
           "--work", os.path.join(work, "scenarios"), "--report", report]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env=env, cwd=work, timeout=90 * 60).returncode
    try:
        with open(report, encoding="utf-8") as f:
            checks = json.load(f)["checks"]
    except (OSError, ValueError, KeyError):
        checks = []
    failed = [c for c in checks if c["status"] == "FAIL"]
    passed = sum(c["status"] == "ok" for c in checks)
    check(rc == 0 and checks and not failed,
          f"app scenarios: {passed} checks passed, {len(failed)} failed (exit code {rc})",
          "; ".join(f"[{c['step']}] {c['check']}" for c in failed))


def launch_like_shortcut(dest, work, version, documents):
    """Open the Start-menu shortcut through the shell, as a click on it does,
    from the naive environment."""
    lnk = os.path.join(START_MENU, "LIMBUS.lnk")
    title = f"Camera Video Recorder — LIMBUS {version}"
    if windows_titled(title):
        check(False, "no LIMBUS window is open before the launch")
        return None
    extra = {} if documents else {"LIMBUS_RECORDINGS": os.path.join(work, "launch_recordings")}
    log_before = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    env, saved = naive_env(**extra), dict(os.environ)
    os.environ.clear()
    os.environ.update(env)                       # what the shell hands the app
    try:
        t0 = time.time()
        os.startfile(lnk)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    found = []
    while time.time() - t0 < 300 and not found:
        found = windows_titled(title)
        time.sleep(0.5)
    if not check(bool(found), f"its window opens ({time.time() - t0:.0f}s): {title!r}"):
        return None
    hwnd, pid = found[0]
    handle = kernel32.OpenProcess(SYNCHRONIZE | QUERY_LIMITED, False, pid)
    try:
        image = process_image(handle)
        check(os.path.normcase(image) == os.path.normcase(os.path.join(dest, "runtime", "pythonw.exe")),
              "it runs on the installed pythonw.exe (no console)", image)
        check(wait_process(handle, 8) is None, "it keeps running")
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        rc = wait_process(handle, 90)
        check(rc == 0, f"it closes cleanly (exit code {rc})")
    finally:
        kernel32.CloseHandle(handle)
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            f.seek(log_before)
            log = f.read()
    except OSError:
        log = ""
    check(f"LIMBUS {version}" in log and "Traceback" not in log,
          f"{LOG_PATH} has its startup line and no errors", log[-1500:])
    if documents:
        docs = os.path.join(os.environ["USERPROFILE"], "Documents")
        found = [d for d in (os.path.join(docs, "LIMBUS", "recordings"),
                             os.path.join(os.environ.get("OneDrive", docs), "Documents", "LIMBUS", "recordings"))
                 if os.path.isdir(d)]
        check(bool(found), "its recordings folder is Documents\\LIMBUS\\recordings", docs)
        return found[0] if found else None
    check(os.path.isdir(extra["LIMBUS_RECORDINGS"]), "its recordings folder was created")
    return extra["LIMBUS_RECORDINGS"]


def uninstall(dest, desktop, kept):
    unins = os.path.join(dest, "unins000.exe")
    subprocess.run([unins, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])
    t0 = time.time()
    # the uninstaller hands over to a copy of itself and returns at once
    while time.time() - t0 < 300 and (os.path.exists(dest) or uninstall_entry() is not None):
        time.sleep(1)
    check(not os.path.exists(dest), f"the installation folder is gone ({time.time() - t0:.0f}s)",
          "; ".join(os.listdir(dest)[:10]) if os.path.exists(dest) else "")
    check(uninstall_entry() is None, "it is gone from Apps & features")
    left = [n for n in ("LIMBUS.lnk", "LIMBUS Self-Test.lnk") if os.path.exists(os.path.join(START_MENU, n))]
    if desktop and os.path.exists(os.path.join(DESKTOP, "LIMBUS.lnk")):
        left.append("Desktop\\LIMBUS.lnk")
    check(not left, "its shortcuts are gone", ", ".join(left))
    if kept:
        check(os.path.isdir(kept), f"the recordings are kept ({kept})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--installer", required=True)
    ap.add_argument("--work", help="working folder (default: a new temporary one)")
    ap.add_argument("--dir", help="install here (default: where the installer installs by "
                                  "default, %%LOCALAPPDATA%%\\Programs\\LIMBUS)")
    ap.add_argument("--documents", action="store_true",
                    help="let the shortcut launch record into the real Documents\\LIMBUS")
    ap.add_argument("--desktop", action="store_true", help="also install the desktop shortcut")
    ap.add_argument("--keep-installed", action="store_true", help="skip the uninstall")
    args = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("the installer is for Windows")
    installer = os.path.abspath(args.installer)
    version = os.path.basename(installer)[len("LIMBUS-Setup-"):-len(".exe")]
    work = os.path.abspath(args.work or tempfile.mkdtemp(prefix="limbus_installer_e2e_"))
    os.makedirs(work, exist_ok=True)
    dest = os.path.abspath(args.dir) if args.dir else None
    default = os.path.join(os.environ["LOCALAPPDATA"], "Programs", "LIMBUS")
    if uninstall_entry() is not None:
        sys.exit(f"LIMBUS is already installed for this user ({uninstall_entry()['InstallLocation']}): "
                 "uninstall it first, so this run can't touch a real installation")
    t0 = time.time()
    print(f"installer {installer} ({os.path.getsize(installer) / 1e6:.0f} MB), version {version}")
    print(f"work folder {work}")
    installed = False
    kept = None
    try:
        section("install")
        refuses_deep_folder(installer, work)
        rc, secs = run_installer(installer, dest, os.path.join(work, "install.log"), args.desktop)
        entry = uninstall_entry()
        if dest is None:
            dest = os.path.normpath((entry or {}).get("InstallLocation") or default)
            check(os.path.normcase(dest) == os.path.normcase(default),
                  f"installed where it installs by default: {dest}", default)
        installed = rc == 0 or os.path.exists(dest)
        if check(rc == 0, f"silent install into {dest}, exit code {rc} ({secs:.0f}s)"):
            guarded(check_install, dest, version, args.desktop)

            section("install again over it (an upgrade)")
            rc, secs = run_installer(installer, dest, os.path.join(work, "upgrade.log"), args.desktop)
            check(rc == 0, f"silent reinstall, exit code {rc} ({secs:.0f}s)")
            guarded(check_install, dest, version, args.desktop)

            section("the app, end to end")
            guarded(run_scenarios, dest, work)

            section("started from its Start-menu shortcut")
            kept = guarded(launch_like_shortcut, dest, work, version, args.documents)
    finally:
        if installed and dest and not args.keep_installed:
            section("uninstall")
            guarded(uninstall, dest, args.desktop, kept)

    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed in {time.time() - t0:.0f}s")
    for r in failed:
        print(f"  FAIL  {r['check']}")
    print("RESULT:", "FAIL" if failed else "PASS", flush=True)
    with open(os.path.join(work, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"result": "FAIL" if failed else "PASS", "installer": installer, "version": version,
                   "checks": results}, f, indent=2)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
