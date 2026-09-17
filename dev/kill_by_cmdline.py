"""按命令行子串查杀 Windows python 进程（避免 git-bash 下的 PowerShell 引号问题）。"""
import subprocess
import sys

needle = sys.argv[1] if len(sys.argv) > 1 else 'sweep_worksplit'
kill = '--kill' in sys.argv

cmd = ['powershell', '-NoProfile', '-Command',
       "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
       "Where-Object { $_.CommandLine -like '*%s*' } | "
       "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation" % needle]

out = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout
lines = out.splitlines()
hits = []
for line in lines[1:]:  # skip header
    if not line.strip():
        continue
    try:
        pid = line.split(',')[0].strip('"')
        if pid.isdigit():
            hits.append((pid, line[:160]))
    except Exception:
        pass

for pid, cmdline in hits:
    print(f'PID {pid}: {cmdline}')
if not hits:
    print(f'no python.exe with {needle!r} in command line')
elif kill:
    for pid, _ in hits:
        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
        print(f'killed {pid}')
