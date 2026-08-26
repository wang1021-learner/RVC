; RVC 实时变声 · 服务器客户端安装包
; 编译: ISCC.exe installer\rvc_server_client.iss

#define MyAppName "RVC实时变声"
#define MyAppPublisher "RVC"
#define MyAppExeName "RVC实时变声.exe"
#define MyAppVersion "1.0.0"

[Setup]
AppId={{8E2C4B1A-7D3F-4A90-9C21-B6E8F0A1D5C3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=RVC实时变声-服务器版安装
SetupIconFile=..\assets\icons\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
ShowLanguageDialog=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Messages]
SetupAppTitle=安装
SetupWindowTitle=安装 {#MyAppName}
UninstallAppTitle=卸载
UninstallAppFullTitle=卸载 {#MyAppName}
ButtonBack=< 上一步(&B)
ButtonNext=下一步(&N) >
ButtonInstall=安装(&I)
ButtonFinish=完成(&F)
ButtonCancel=取消
ButtonYes=是(&Y)
ButtonNo=否(&N)
ButtonNewFolder=新建文件夹(&N)
ClickNext=单击「下一步」继续。
WelcomeLabel1=欢迎安装 [name]
WelcomeLabel2=即将在您的电脑上安装 [name/ver]。#n#n建议先关闭其它应用程序。
SelectDirLabel3=程序将安装到下面的文件夹。
SelectDirBrowseLabel=单击「下一步」继续。若要选择其它文件夹，请单击「浏览」。
DiskSpaceGBLabel=至少需要 [gb] GB 可用空间。
DiskSpaceMBLabel=至少需要 [mb] MB 可用空间。
ReadyLabel1=安装程序已准备好开始安装 [name]。
ReadyLabel2a=单击「安装」继续。若要复查或更改设置，请单击「上一步」。
InstallingLabel=正在安装 [name]，请稍候。
FinishedHeadingLabel=正在完成 [name] 安装向导
FinishedLabelNoIcons=已完成 [name] 的安装。
FinishedLabel=已完成 [name] 的安装。可以通过开始菜单或桌面快捷方式启动。
ClickFinish=单击「完成」退出安装向导。
SelectTasksLabel2=请选择要执行的附加任务，然后单击「下一步」。
WizardSelectDir=选择安装位置
WizardReady=准备安装
WizardInstalling=正在安装
WizardFinished=安装完成
StatusCreateDirs=正在创建目录...
StatusExtractFiles=正在复制文件...
StatusCreateIcons=正在创建快捷方式...
ConfirmUninstall=确定要完全移除 %1 及其所有组件吗？
UninstalledAll=%1 已成功卸载。

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "..\dist\RVC服务器客户端\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\RVC服务器客户端\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\RVC服务器客户端\pack_mode.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\RVC服务器客户端\使用说明.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.txt"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即运行 {#MyAppName}"; Flags: nowait postinstall skipifsilent
