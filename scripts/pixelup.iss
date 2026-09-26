; Inno Setup script — builds dist\pixelup-<version>-setup.exe from the PyInstaller
; onedir output in dist\PixelUp\. The version is passed in by scripts/package.ps1
; via /DMyAppVersion. iscc is pre-installed on windows-latest.

#define MyAppName "PixelUp"
#define MyAppPublisher "Yoshinao Inoguchi"
#define MyAppExe "PixelUp.exe"
#ifndef MyAppVersion
  #error MyAppVersion is not defined - pass it via  iscc /DMyAppVersion=x.y.z
#endif

[Setup]
; This .iss lives in scripts/, but the PyInstaller output and the dist/ output folder
; are at the repo root — so resolve all source/output paths one level up.
SourceDir=..
AppName={#MyAppName}
AppId={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExe}
Uninstallable=yes
OutputDir=dist
OutputBaseFilename=pixelup-{#MyAppVersion}-setup
SetupIconFile=build\icon.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequiredOverridesAllowed=dialog
; The wizard speaks the Windows display language when Inno ships it, and asks
; only when it does not (localization-conventions).
ShowLanguageDialog=auto

[Languages]
; Inno's own built-in translations of the interface languages, English first as
; the fallback. Inno ships none for Korean or Simplified Chinese, and no
; third-party .isl files are vendored, so those readers get the English wizard.
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "dist\PixelUp\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Tasks]
; The custom strings are Inno's own standard messages, translated in every
; language above.
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
; Inno cannot recover a non-elevated user token for every elevated setup path.
; All-users installs launch later through their scoped shell shortcuts.
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent runasoriginaluser; Check: not IsAdminInstallMode
