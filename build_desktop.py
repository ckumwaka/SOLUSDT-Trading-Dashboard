"""
Desktop application packaging script using PyInstaller
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name}")

def create_spec_file():
    """Create PyInstaller spec file for advanced configuration"""[25][28][30]
    
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('config.py', '.'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'trading_logic',
        'feature_engineering', 
        'ml_model',
        'data_manager',
        'websocket_manager',
        'sklearn.ensemble',
        'sklearn.tree._utils',
        'numba',
        'polars',
        'talib',
        'webview',
        'flask',
        'numpy',
        'pandas'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'plotly',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SOL_Trading_Dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
    version='version_info.txt' if os.path.exists('version_info.txt') else None
)
"""
    
    with open('sol_trading.spec', 'w') as f:
        f.write(spec_content)
    
    print("Created sol_trading.spec file")

def create_version_info():
    """Create version info file for Windows executable"""
    
    version_info = """
# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'SOL Trading Solutions'),
        StringStruct(u'FileDescription', u'Professional SOL/USDT Trading Dashboard'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'SOL_Trading_Dashboard'),
        StringStruct(u'LegalCopyright', u'Copyright © 2025'),
        StringStruct(u'OriginalFilename', u'SOL_Trading_Dashboard.exe'),
        StringStruct(u'ProductName', u'SOL Trading Dashboard'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    
    with open('version_info.txt', 'w') as f:
        f.write(version_info)
    
    print("Created version_info.txt")

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'flask', 'pywebview', 'pandas', 'numpy', 'scikit-learn',
        'joblib', 'requests', 'websocket-client', 'numba', 'polars'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("All dependencies are available")
    return True

def build_application():
    """Build the desktop application"""[25][28]
    
    try:
        print("Building SOL Trading Dashboard...")
        
        # Build using spec file
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            'sol_trading.spec',
            '--clean',
            '--noconfirm'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        print("Build completed successfully!")
        
        # Check if executable was created
        exe_path = Path('dist/SOL_Trading_Dashboard.exe')
        if exe_path.exists():
            print(f"Executable created: {exe_path.absolute()}")
            print(f"File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            return True
        else:
            print("Executable not found in dist directory")
            return False
            
    except Exception as e:
        print(f"Build error: {e}")
        return False

def create_installer_script():
    """Create NSIS installer script for Windows"""
    
    nsis_script = """
!define APPNAME "SOL Trading Dashboard"
!define COMPANYNAME "SOL Trading Solutions"
!define DESCRIPTION "Professional cryptocurrency trading analysis"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/your-repo/sol-trading"
!define UPDATEURL "https://github.com/your-repo/sol-trading/releases"
!define ABOUTURL "https://github.com/your-repo/sol-trading"
!define INSTALLSIZE 150000

RequestExecutionLevel admin

InstallDir "$PROGRAMFILES\\${COMPANYNAME}\\${APPNAME}"
LicenseData "LICENSE"
Name "${APPNAME}"
Icon "icon.ico"
outFile "SOL_Trading_Dashboard_Installer.exe"

!include LogicLib.nsh

page license
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    file "dist\\SOL_Trading_Dashboard.exe"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${COMPANYNAME}"
    createShortCut "$SMPROGRAMS\\${COMPANYNAME}\\${APPNAME}.lnk" "$INSTDIR\\SOL_Trading_Dashboard.exe"
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\SOL_Trading_Dashboard.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "InstallLocation" "$\\"$INSTDIR$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "DisplayIcon" "$\\"$INSTDIR\\SOL_Trading_Dashboard.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "HelpLink" "${HELPURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\SOL_Trading_Dashboard.exe"
    delete "$INSTDIR\\uninstall.exe"
    rmDir "$INSTDIR"
    
    delete "$SMPROGRAMS\\${COMPANYNAME}\\${APPNAME}.lnk"
    rmDir "$SMPROGRAMS\\${COMPANYNAME}"
    delete "$DESKTOP\\${APPNAME}.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${COMPANYNAME} ${APPNAME}"
sectionEnd
"""
    
    with open('installer.nsi', 'w') as f:
        f.write(nsis_script)
    
    print("Created installer.nsi script")

def main():
    """Main build process"""
    
    print("SOL Trading Dashboard - Build Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Clean previous builds
    clean_build_dirs()
    
    # Create build files
    create_spec_file()
    create_version_info()
    create_installer_script()
    
    # Build application
    if build_application():
        print("\n" + "=" * 50)
        print("Build completed successfully!")
        print("Executable location: dist/SOL_Trading_Dashboard.exe")
        print("\nTo create an installer:")
        print("1. Install NSIS (https://nsis.sourceforge.io/)")
        print("2. Run: makensis installer.nsi")
        print("\nTo run the application:")
        print("Double-click SOL_Trading_Dashboard.exe")
        return True
    else:
        print("Build failed!")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
