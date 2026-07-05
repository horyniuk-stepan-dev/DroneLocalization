import sys, os, ctypes

print('--- DIAGNOSTIC SCRIPT START ---')
base = r'd:\My Projects\DroneLocalization\dist\DroneLocalization\_internal'
torch_lib = os.path.join(base, 'torch', 'lib')

os.add_dll_directory(base)
os.add_dll_directory(torch_lib)

kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
kernel32.LoadLibraryExW.restype = ctypes.c_void_p

deps = ['VCRUNTIME140.dll', 'MSVCP140.dll', 'c10.dll', 'c10_cuda.dll']
for dep in deps:
    try:
        if dep.startswith('c1'):
            path = os.path.join(torch_lib, dep)
        else:
            path = dep
        
        # 0x00001100 = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
        res = kernel32.LoadLibraryExW(path, None, 0x00001100)
        if res is None:
            err = ctypes.WinError(ctypes.get_last_error())
            print(f'FAIL: {dep} - {err}')
        else:
            print(f'OK: {dep}')
    except Exception as e:
        print(f'FAIL EXCEPTION: {dep} - {e}')
print('--- DIAGNOSTIC SCRIPT END ---')
