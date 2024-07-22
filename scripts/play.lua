---@diagnostic disable: undefined-global

print("Loaded script!")
savestate.load("../savestates/start.State")

comm.socketServerSend("Connected!")


while true do
    -- print(mainmemory.readbyte(0x0010DC)) --checkpoint 
    
    comm.socketServerScreenShot()

    print(comm.socketServerResponse())
    emu.frameadvance()
end
