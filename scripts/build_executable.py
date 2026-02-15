#!/usr/bin/env python3
"""
Build executable using PyInstaller
"""

import subprocess
import sys
from pathlib import Path


def build_executable():
    """Build executable with PyInstaller"""
    
    spec_file = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../data/models', 'data/models'),
        ('../config', 'config'),
        ('../src/gui/resources', 'src/gui/resources'),
    ],
    hiddenimports=[
        'PyQt6',
        'torch',
        'ultralytics',
        'filterpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DroneLocalizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DroneLocalizer'
)
"""
    
    # Write spec file
    spec_path = Path("DroneLocalizer.spec")
    spec_path.write_text(spec_file)
    
    print("Building executable...")
    # TODO: Run PyInstaller
    # subprocess.run([sys.executable, "-m", "PyInstaller", str(spec_path)])
    
    print("Build complete! Check dist/ folder")


if __name__ == "__main__":
    build_executable()
