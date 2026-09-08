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
**Make sure to Configure the Launch settings in the `settings.cfg` file first**

 - `EMUHAWK_FILEPATH` - File path to the Bizhawk emulator
 - `ROM_FILEPATH` - File path to the Mario kart ROM (I used the USA version, but it should not matter). **ROM IS NOT PROVIDED**
 - `SOCKET_IP` - IP where the emulator is launched. *Leave it as localhost.* Since the application uses sockets, it has the potential of having the emulator and model on different computers. You would need to change the run file significantly and this project currently does not support it. 
 - `SOCKET_PORT` - Port(s) where the emulator is launched. If the `NUM_ENV` is greater than 1, then the ports will be increments of that. Eg. `SOCKET_PORT = 8080` and `NUM_ENV = 3`, then ports 8080, 8081 and 8082 will be used.
 - `NUM_ENV` - Number of enviroments to be launched. Each enviroment will launch a different instance of Bizhawk.


To get started on training just use python to run the `run.py` file. It will automatically load the `settings.cfg` file with the hyperparameters specified.

```bash
python run.py
```

## Design

### Model Architecture


```
    200 × 66 × 3 RGB image
             │
             ▼
┌──────────────────────────┐
│       CNN Encoder        │
│                          │
│  3 → 24   Conv 5×5 / 2   │
│ 24 → 36   Conv 5×5 / 2   │
│ 36 → 48   Conv 5×5 / 2   │
│ 48 → 64   Conv 3×3       │
│ 64 → 64   Conv 3×3       │
└────────────┬─────────────┘
             │
             ▼
      ┌─────────────┐
      │  FC Layers  │
      │ 1152 → 256  │
      │  256 → 128  │
      │  128 →   3  │
      └──────┬──────┘
             │
             ▼
        3 Actions
   Steering · Acceleration · Brake
```



Currently the best training parameters that yielded the best results are in `settings.cfg`. Took approximately 400k episodes (~72 hours) of training.
All of this has created a model that can obtain 1st place consistently.


## Improvements
 - Switch to a RNN instead of a CNN so the model as a sense of time and speed.
 - Test on Linux and add a headless mode, so it can be run on cheap cloud servers
 - Better tensorboard naming scheme and test handling
 - Versus mode allowing you to play against the AI
 - Proper Evaluate mode from command line