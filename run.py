import subprocess

EMUHAWK_PATH = "C:/Dev/Bizhawk/EmuHawk.exe"
ROM_PATH = "Super Mario Kart (USA).sfc"

subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/client.lua")

