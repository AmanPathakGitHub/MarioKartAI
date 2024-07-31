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
    
   
   
ports = PORT.split(",")  


for port in ports:
    subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/play.lua --socket_ip={IP_ADDRESS} --socket_port={port}")
# subprocess.Popen(f"{EMUHAWK_PATH} \"{ROM_PATH}\" --lua=scripts/play.lua")
agent = Agent(IP_ADDRESS, ports, config)
agent.run()