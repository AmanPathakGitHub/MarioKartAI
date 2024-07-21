---@diagnostic disable: undefined-global

print("Loaded script!")

savestate.load("../savestates/start.State")

while true do
    print(mainmemory.readbyte(0x0010DC)) --checkpoint 
    

    emu.frameadvance()
end
