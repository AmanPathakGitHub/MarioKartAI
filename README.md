# Mario Kart AI

### Shortcuts
[Tools](#tools)	

[Design](#design)

[Results](#results)

[How to Run](#how-to-run)

## Tools

 - Pytorch
 - Bizhawk Emulator

## How to run 

**Make sure to Configure the Launch settings in the settings.cfg file first**

 - `EMUHAWK_FILEPATH` - File path to the Bizhawk emulator
 - `ROM_FILEPATH` - File path to the Mario kart ROM (I used the USA version, but it should not matter) **ROM IS NOT PROVIDED**
 - `SOCKET_IP` - IP where the emulator is launched. *Leave it as localhost.* Since the application uses sockets, it has the potential of having the emulator and model on different computers. You would need to change the run file significantly and this project currently does not support it. 
 - `SOCKET_PORT` - Port(s) where the emulator is launched. If the `NUM_ENV` is greater than 1, then the ports will be increments of that. Eg. `SOCKET_PORT = 8080` and `NUM_ENV = 3`, then ports 8080, 8081 and 8082 will be used.
 - `NUM_ENV` - 


To get started on training just use python to run the `run.py` file

```bash
python run.py
```


## Design

## Results

im working on it
