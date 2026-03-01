; Скрипт створення інсталятора для Drone Topometric Localization
; Компілюється через Inno Setup

#define MyAppName "Drone Localization"
#define MyAppVersion "1.0"
#define MyAppPublisher "UAV Systems"
#define MyAppExeName "DroneLocalization.exe"

[Setup]
; Унікальний ідентифікатор програми (не змінюйте його для майбутніх оновлень)
AppId={{8A4F1E93-9C12-4B52-A9F3-8D2A1F8C4A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Тека встановлення за замовчуванням (Program Files)
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; Назва вихідного файлу інсталятора
OutputBaseFilename=Install_DroneLocalization
; Директорія, куди збережеться готовий Setup.exe
OutputDir=dist
; Максимальний рівень стиснення (ідеально для важких нейромереж)
Compression=lzma2/ultra64
SolidCompression=yes
; Налаштування зовнішнього вигляду
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Копіюємо всі файли з нашої зібраної папки
Source: "dist\DroneLocalization\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\DroneLocalization\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; УВАГА: Не використовуйте прапорець "ignoreversion" для системних файлів, які вже можуть бути в Windows, але тут це безпечно, бо ми ставимо в ізольовану теку.

[Icons]
; Створюємо ярлики в меню Пуск та на робочому столі
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Пропонуємо запустити програму після завершення встановлення
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent