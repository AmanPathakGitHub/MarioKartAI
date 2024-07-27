import subprocess
import configparser

from src.agent import Agent

with open("settings.cfg") as file:
    config = configparser.ConfigParser()
    config.read_file(file)
    EMUHAWK_PATH = config.get("Launch", "EMUHAWK_FILEPATH")
    ROM_PATH = config.get("Launch", "ROM_FILEPATH")
    IP_ADDRESS = config.get("Launch", "SOCKET_IP")
    PORT = config.get("Launch", "SOCKET_PORT")
    
    MAX_STEPS = config.get("Training", "MAX_STEPS")
    REPLAY_MEMORY_SIZE = config.get("Training", "REPLAY_MEMORY_SIZE")


subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/play.lua --socket_ip={IP_ADDRESS} --socket_port={PORT}")
# subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/play.lua")
agent = Agent(IP_ADDRESS, PORT, MAX_STEPS, REPLAY_MEMORY_SIZE)
agent.run()