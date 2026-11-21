"""Foot pedal: hold it down to record.

USB foot pedals such as the PCsensor FootSwitch present themselves as a
keyboard and type a key while pressed (out of the box, the letter "b").
LIMBUS recognizes the pedal by its USB ID through Windows Raw Input, not by
the key it types, so

  - the keyboard's own "b" never starts a recording,
  - the pedal works whatever key it has been programmed to send,
  - it works while another window has the focus, and
  - the keystrokes it types are kept out of LIMBUS's own fields.

Windows only: elsewhere FootPedal.available is False and start() does
nothing. Set LIMBUS_PEDAL_DEBUG=1 to print what the pedal sends.
"""

import collections
import ctypes
import os
import sys
import time

from PyQt5 import QtCore

# USB vendor and product IDs of foot pedals, as they appear in the device
# path Windows reports. Any key such a device sends counts as the pedal.
PEDAL_IDS = (
    "VID_3553&PID_B001",    # PCsensor FootSwitch (current models)
    "VID_0C45&PID_7403",    # PCsensor FootSwitch (older models)
    "VID_0C45&PID_7404",
    "VID_413D&PID_2107",
    "VID_1A86&PID_E026",    # iKKEGOL
)

DEBUG = bool(os.environ.get("LIMBUS_PEDAL_DEBUG"))

# Each keystroke the pedal types reaches the app twice: first as raw input,
# then as the ordinary key message it turns into. That second copy is
# dropped if it arrives within this long of the first.
SWALLOW_WINDOW_S = 0.5

if sys.platform == "win32":
    from ctypes import wintypes

    # a private handle, so these prototypes don't leak into other code
    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    WM_INPUT_DEVICE_CHANGE = 0x00FE
    WM_INPUT = 0x00FF
    RIDEV_REMOVE = 0x00000001
    RIDEV_INPUTSINK = 0x00000100     # also while another window has the focus
    RIDEV_DEVNOTIFY = 0x00002000     # tell us when keyboards come and go
    RID_INPUT = 0x10000003
    RIDI_DEVICENAME = 0x20000007
    RIM_TYPEKEYBOARD = 1
    RI_KEY_BREAK = 0x01
    GIDC_ARRIVAL, GIDC_REMOVAL = 1, 2
    FAILED = 0xFFFFFFFF              # (UINT)-1, the raw input calls' error value
    PTR_MASK = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
    MSG_OFFSET = ctypes.sizeof(ctypes.c_void_p)   # MSG.message follows MSG.hwnd

    class RAWINPUTDEVICE(ctypes.Structure):
        _fields_ = [("usUsagePage", wintypes.USHORT),
                    ("usUsage", wintypes.USHORT),
                    ("dwFlags", wintypes.DWORD),
                    ("hwndTarget", wintypes.HWND)]

    class RAWINPUTDEVICELIST(ctypes.Structure):
        _fields_ = [("hDevice", wintypes.HANDLE), ("dwType", wintypes.DWORD)]

    class RAWINPUTHEADER(ctypes.Structure):
        _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                    ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]

    class RAWKEYBOARD(ctypes.Structure):
        _fields_ = [("MakeCode", wintypes.USHORT), ("Flags", wintypes.USHORT),
                    ("Reserved", wintypes.USHORT), ("VKey", wintypes.USHORT),
                    ("Message", wintypes.UINT),
                    ("ExtraInformation", wintypes.ULONG)]

    class RAWINPUT(ctypes.Structure):
        # only keyboards are registered, so only this member of the union
        _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]

    _user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
    _user32.RegisterRawInputDevices.restype = wintypes.BOOL
    _user32.GetRawInputData.argtypes = [
        wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
        ctypes.POINTER(wintypes.UINT), wintypes.UINT]
    _user32.GetRawInputData.restype = wintypes.UINT
    _user32.GetRawInputDeviceInfoW.argtypes = [
        wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
        ctypes.POINTER(wintypes.UINT)]
    _user32.GetRawInputDeviceInfoW.restype = wintypes.UINT
    _user32.GetRawInputDeviceList.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICELIST), ctypes.POINTER(wintypes.UINT),
        wintypes.UINT]
    _user32.GetRawInputDeviceList.restype = wintypes.UINT

    def device_name(handle):
        """The device path Windows gives a raw input device ("" if gone)."""
        size = wintypes.UINT(0)
        _user32.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, None,
                                       ctypes.byref(size))
        if not size.value:
            return ""
        buf = ctypes.create_unicode_buffer(size.value + 1)
        if _user32.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, buf,
                                          ctypes.byref(size)) in (0, FAILED):
            return ""
        return buf.value

    def list_pedals():
        """Device handles of the foot pedals plugged in now."""
        item = ctypes.sizeof(RAWINPUTDEVICELIST)
        count = wintypes.UINT(0)
        if _user32.GetRawInputDeviceList(None, ctypes.byref(count),
                                         item) == FAILED:
            return set()
        devices = (RAWINPUTDEVICELIST * max(count.value, 1))()
        n = _user32.GetRawInputDeviceList(devices, ctypes.byref(count), item)
        if n == FAILED:
            return set()
        return {int(d.hDevice) for d in devices[:n]
                if d.dwType == RIM_TYPEKEYBOARD and d.hDevice
                and is_pedal_name(device_name(d.hDevice))}


def is_pedal_name(name):
    name = (name or "").upper()
    return any(ids in name for ids in PEDAL_IDS)


_KEY_EVENTS = {QtCore.QEvent.ShortcutOverride: False,
               QtCore.QEvent.KeyPress: False,
               QtCore.QEvent.KeyRelease: True}       # event -> is a release


class _NativeFilter(QtCore.QAbstractNativeEventFilter):
    """Hands every Windows message the app sees to FootPedal._filter."""

    def __init__(self, pedal):
        super().__init__()
        self.pedal = pedal
        self.failed = False

    def nativeEventFilter(self, event_type, message):
        try:
            return self.pedal._filter(bytes(event_type), int(message)), 0
        except Exception as exc:
            if not self.failed:      # once, not for every message after
                self.failed = True
                print(f"foot pedal: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            return False, 0


class FootPedal(QtCore.QObject):
    """Tells when a USB foot pedal goes down and when it comes back up.

    pressed and released fire once per press however long the pedal is held:
    the key repeats it sends meanwhile are ignored. present_changed fires as
    a pedal is plugged in or unplugged. All three are delivered from the
    event loop, never from inside the Windows message being handled."""

    pressed = QtCore.pyqtSignal()
    released = QtCore.pyqtSignal()
    present_changed = QtCore.pyqtSignal(bool)

    available = sys.platform == "win32"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hwnd = None
        self._native = None
        self._pedals = set()     # device handles of the pedals plugged in
        self._known = {}         # device handle -> is it a pedal
        self._held = set()       # (device, key) pairs held down right now
        # pedal keystrokes whose ordinary key message is still to come:
        # (virtual key, is a release, when)
        self._pending = collections.deque(maxlen=64)
        if self.available:
            self._buf = (ctypes.c_byte * 64)()
            self._raw = RAWINPUT.from_buffer(self._buf)

    @property
    def present(self):
        return bool(self._pedals)

    @property
    def held(self):
        return bool(self._held)

    @property
    def listening(self):
        return self._hwnd is not None

    def start(self, window):
        """Listen for pedals. window is a top-level widget: Windows delivers
        the keyboards' raw input to it, focused or not."""
        if not self.available or self._hwnd is not None:
            return False
        hwnd = int(window.winId())
        dev = RAWINPUTDEVICE(0x01, 0x06,          # generic desktop: keyboard
                             RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, hwnd)
        if not _user32.RegisterRawInputDevices(ctypes.byref(dev), 1,
                                               ctypes.sizeof(dev)):
            print("foot pedal: could not listen for keyboards "
                  f"(error {ctypes.get_last_error()})", file=sys.stderr)
            return False
        self._hwnd = hwnd
        self._native = _NativeFilter(self)
        app = QtCore.QCoreApplication.instance()
        app.installNativeEventFilter(self._native)
        app.installEventFilter(self)      # to drop the keys the pedal types
        self._pedals = list_pedals()
        self._known = {h: True for h in self._pedals}
        if DEBUG:
            print(f"foot pedal: listening, {len(self._pedals)} pedal(s) "
                  "plugged in", flush=True)
        if self._pedals:
            self._later(self.present_changed.emit, True)
        return True

    def stop(self):
        if self._hwnd is None:
            return
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
            if self._native is not None:
                app.removeNativeEventFilter(self._native)
        dev = RAWINPUTDEVICE(0x01, 0x06, RIDEV_REMOVE, None)
        _user32.RegisterRawInputDevices(ctypes.byref(dev), 1,
                                        ctypes.sizeof(dev))
        self._hwnd = self._native = None
        self._held.clear()
        self._pending.clear()

    # ---- Windows messages and Qt key events ---------------------------------
    def _filter(self, event_type, addr):
        """Read raw input where Qt hands our window its messages, so it is
        seen even inside a native modal loop such as a window resize. The
        message itself always goes on to Windows' default handling."""
        if event_type != b"windows_generic_MSG":
            return False
        message = ctypes.c_uint.from_address(addr + MSG_OFFSET).value
        if message == WM_INPUT:
            msg = wintypes.MSG.from_address(addr)
            self._on_input(msg.lParam & PTR_MASK)
        elif message == WM_INPUT_DEVICE_CHANGE:
            msg = wintypes.MSG.from_address(addr)
            self._on_device_change(msg.wParam, msg.lParam & PTR_MASK)
        return False

    def eventFilter(self, obj, event):
        """Drop the key events a pedal keystroke turned into. Windows
        delivers a keystroke's raw input before its key message, so by the
        time Qt makes a key event of it, _pending already holds it."""
        if not self._pending:
            return False
        up = _KEY_EVENTS.get(event.type())
        if up is None:
            return False
        if (event.type() == QtCore.QEvent.ShortcutOverride
                or (up and event.isAutoRepeat())):
            # Both come just ahead of a KeyPress for the same keystroke: a
            # ShortcutOverride (claimed, so the key can't fire a shortcut),
            # and the release Qt makes up before each auto-repeat press.
            # Drop them, but leave the keystroke pending for that KeyPress.
            if self._matches(event.nativeVirtualKey(), False) is not None:
                event.accept()
                return True
            return False
        if self._swallow(event.nativeVirtualKey(), up):
            event.accept()
            return True
        return False

    def _on_input(self, hraw):
        size = wintypes.UINT(ctypes.sizeof(self._buf))
        n = _user32.GetRawInputData(hraw, RID_INPUT, self._buf,
                                    ctypes.byref(size),
                                    ctypes.sizeof(RAWINPUTHEADER))
        if n == FAILED or n < ctypes.sizeof(RAWINPUT):
            return
        raw = self._raw
        if raw.header.dwType != RIM_TYPEKEYBOARD:
            return
        device = int(raw.header.hDevice or 0)
        if not self._is_pedal(device):
            return
        kb = raw.keyboard
        self._key(device, kb.VKey, bool(kb.Flags & RI_KEY_BREAK))

    def _key(self, device, vkey, up):
        """One keystroke from a pedal: a press, a repeat or a release."""
        self._pending.append((vkey, up, time.monotonic()))
        key = (device, vkey)
        if up:
            if key in self._held:
                self._held.discard(key)
                if not self._held:
                    self._later(self.released.emit)
        elif key not in self._held:
            first = not self._held
            self._held.add(key)
            if first:
                self._later(self.pressed.emit)
        if DEBUG:
            print(f"foot pedal: key 0x{vkey:02X} {'up' if up else 'down'}, "
                  f"held={self.held}", flush=True)

    def _is_pedal(self, device):
        if not device:
            return False         # injected input (SendInput) has no device
        known = self._known.get(device)
        if known is None:
            known = self._known[device] = is_pedal_name(device_name(device))
            if known:
                self._add(device)
        return known

    def _on_device_change(self, kind, device):
        if kind == GIDC_ARRIVAL:
            pedal = self._known[device] = is_pedal_name(device_name(device))
            if pedal:
                self._add(device)
        elif kind == GIDC_REMOVAL:
            self._known.pop(device, None)
            if device not in self._pedals:
                return
            self._pedals.discard(device)
            # unplugged mid-press: its release will never come
            was_held = self.held
            self._held = {k for k in self._held if k[0] != device}
            if was_held and not self._held:
                self._later(self.released.emit)
            if DEBUG:
                print("foot pedal: unplugged", flush=True)
            if not self._pedals:
                self._later(self.present_changed.emit, False)

    def _add(self, device):
        if device in self._pedals:
            return
        first = not self._pedals
        self._pedals.add(device)
        if DEBUG:
            print("foot pedal: plugged in", flush=True)
        if first:
            self._later(self.present_changed.emit, True)

    def _matches(self, vkey, up):
        """Index in _pending of a pedal keystroke for this key, or None."""
        now = time.monotonic()
        while self._pending and now - self._pending[0][2] > SWALLOW_WINDOW_S:
            self._pending.popleft()
        for i, (k, u, _) in enumerate(self._pending):
            if k == vkey and u == up:
                return i
        return None

    def _swallow(self, vkey, up):
        """True (and forget it) if a pedal keystroke for this key is
        pending: the key event is the pedal's, not the keyboard's."""
        i = self._matches(vkey, up)
        if i is None:
            return False
        del self._pending[i]
        return True

    @staticmethod
    def _later(fn, *args):
        QtCore.QTimer.singleShot(0, lambda: fn(*args))
