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
    NUM_ENV = int(config.get("Launch", "NUM_ENV"))
    
    CHECKPOINT_PATH = config.get("Checkpoint", "FILE_PATH")


for i in range(NUM_ENV):
    subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/play.lua --socket_ip={IP_ADDRESS} --socket_port={str(int(PORT)+i)}")

agent = Agent(IP_ADDRESS, PORT, NUM_ENV, config)
agent.loadCheckpoint(CHECKPOINT_PATH)
agent.run()