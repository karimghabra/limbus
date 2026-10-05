; The LIMBUS installer: packs build\bundle (made by installer\build_bundle.py)
; into a single dist\LIMBUS-Setup-<version>.exe. Build it with Inno Setup 6:
;
;     iscc installer\limbus.iss                      (version from VERSION)
;     iscc /DAppVersion=0.1.0 installer\limbus.iss
;
; The setup needs no administrator rights: it installs for the current user
; (%LOCALAPPDATA%\Programs\LIMBUS), or for everyone if the user chooses so.
; It adds Start-menu shortcuts (LIMBUS, and LIMBUS Self-Test, which runs
; e2e\app_scenarios.py against the installed copy) and, optionally, one on
; the desktop. The app keeps its recordings in Documents\LIMBUS, which an
; uninstall leaves alone. Silent install for scripts and CI:
;
;     LIMBUS-Setup-0.1.0.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /DIR=C:\somewhere

#ifndef AppVersion
  #define VersionFile FileOpen(AddBackslash(SourcePath) + "..\VERSION")
  #define AppVersion Trim(FileRead(VersionFile))
  #expr FileClose(VersionFile)
#endif
#ifndef BundleDir
  #define BundleDir AddBackslash(SourcePath) + "..\build\bundle"
#endif
; the longest file path inside the bundle, relative to the install folder
; (build_bundle.py measures it): the folder must leave room for it
#define LongestPath ReadIni(BundleDir + ".ini", "bundle", "longest_path", "150")
#define AppName "LIMBUS"
; the same ID as APP_USER_MODEL_ID in camera_recorder.py: the taskbar then
; treats the running app and its shortcuts as one
#define AppUserModelID "LIMBUS.CameraRecorder"
; (single-quoted for ISPP, so the doubled quotes reach the [Icons] lines as is)
#define AppArgs '-s ""{app}\app\camera_recorder.py""'

[Setup]
; never change AppId: upgrades find the existing install by it
AppId={{12D5E0C6-4707-412F-9370-4E23C3BFA622}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=LIMBUS
AppPublisherURL=https://github.com/karimghabra/limbus
AppSupportURL=https://github.com/karimghabra/limbus/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#SourcePath}\..\dist
OutputBaseFilename=LIMBUS-Setup-{#AppVersion}
SetupIconFile={#SourcePath}\..\assets\limbus.ico
UninstallDisplayIcon={app}\app\assets\limbus.ico
UninstallDisplayName={#AppName} {#AppVersion}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
; an upgrade first closes a running copy (it holds files in {app}\runtime)
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; an upgrade replaces the Python runtime and the app wholesale, so no
; package or file of the previous version is left behind
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#AppArgs}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\limbus.ico"; Comment: "Record and review the conjunctival microcirculation"; AppUserModelID: "{#AppUserModelID}"
Name: "{autoprograms}\{#AppName} Self-Test"; Filename: "{app}\runtime\python.exe"; Parameters: "-s ""{app}\app\e2e\app_scenarios.py"" --pause"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\limbus.ico"; Comment: "Check this installation end to end (and the camera, if one is plugged in)"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#AppArgs}"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\limbus.ico"; AppUserModelID: "{#AppUserModelID}"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "{#AppArgs}"; WorkingDir: "{app}\app"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; what the app itself wrote under its folder (bytecode caches); recordings
; live in Documents\LIMBUS and are kept
Type: filesandordirs; Name: "{app}"

[Code]
const
  // Windows' limit on a file path (MAX_PATH less its terminating null)
  MaxPathLength = 259;
  LongestPath = {#LongestPath};

// A folder so deep that the bundle's longest path would pass the limit
// can't hold the app: refuse it before anything is copied (silent installs
// too, where Setup "clicks" Next itself and then exits).
function NextButtonClick(CurPageID: Integer): Boolean;
var
  Msg: String;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and
     (Length(AddBackslash(WizardDirValue)) + LongestPath > MaxPathLength) then
  begin
    Msg := 'That folder is too deep: some of LIMBUS''s files would pass Windows'' ' +
           'limit of 260 characters on a file path. Choose a folder whose path is at most ' +
           IntToStr(MaxPathLength - LongestPath - 1) + ' characters long.';
    Log(Msg);
    SuppressibleMsgBox(Msg, mbError, MB_OK, IDOK);
    Result := False;
  end;
end;
