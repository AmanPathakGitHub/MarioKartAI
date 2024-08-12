---@diagnostic disable: undefined-global

print("Loaded script!")
savestate.load("../savestates/start.State")

comm.socketServerSend("Connected!")

local maxstepsmsg = comm.socketServerResponse()

local MAX_STEPS = tonumber(maxstepsmsg)


local prev_checkpoint = 0
local prev_lap = 0

function calculate_reward()
    local checkpoint = mainmemory.readbyte(0x0010DC) 
    local lap = mainmemory.readbyte(0x0010C1)-128
    local lapsize = mainmemory.readbyte(0x000148)

    local ground = mainmemory.readbyte(0x0010AE)
    local collision = mainmemory.readbyte(0x001052)

    local speed = mainmemory.read_u16_be(0x0010EA) --600+ 400-

    local reward = 0

    checkpoint = checkpoint + lap * lapsize

    -- if speed > 500 then 
    --     reward = reward + 1
    -- else
    --     reward = reward - 2
    -- end

    -- if ground == 84 then
    --     reward = reward - 5
    -- end

    if collision > 0 then
        reward = reward - 8
    end

    if prev_checkpoint < checkpoint then
        reward = reward + 10
    elseif prev_checkpoint > checkpoint then
        reward = reward - 10
    end
    
    -- if prev_lap > lap then
    --     reward = reward + 15
        
    -- end

    -- if lap == 133 then
    --     reward = reward + 5
    -- end


    -- sanity check
    -- this is definitely redundant, ill keep it here in case the reward function ever changes
    if reward > 10 then
        reward = 10
    elseif reward < -10 then
        reward = -10
    end
    
    prev_checkpoint = checkpoint
    prev_lap = lap

    return reward
end

-- bad termination
local speed_timer = 0
local checkpoint_timer = 0
function calulate_termination(step)

    
    local speed = mainmemory.read_u16_be(0x0010EA) --600+ 400-
    local checkpoint = mainmemory.readbyte(0x0010DC) 

    if speed < 400 then
        speed_timer = speed_timer + 1
    else
        speed_timer = 0
    end

    if prev_checkpoint == checkpoint or prev_checkpoint > checkpoint then 
        checkpoint_timer = checkpoint_timer + 1
    else
        checkpoint_timer = 0
    end

    if speed_timer > 1000 then
        speed_timer = 0
        return true
    end

    -- Equates to 10 seconds
    -- 60 frames a second, each step is 10 frames 
    if checkpoint_timer > 60 then
        checkpoint_timer = 0
        return true
    end

    if step > MAX_STEPS then
        return true
    end

    return false
end

function bool_to_number(value)
    return value and 1 or 0
end

function recieveActions()
    local action = comm.socketServerResponse()
    action = tonumber(action)
    local INPUTS = {Left=0, Right=0, B=1}

    if action == 0 then INPUTS.Left = 1
    elseif action == 1 then INPUTS.Right = 1
    end

    return INPUTS
end

-- get observation, action, reward and next state


function run()

    while true do
        local step = 0
        local termination = false
        comm.socketServerScreenShot()

        while not termination do
            
            local actions = recieveActions()
            
            for i=0, 10, 1 do
                joypad.set(actions, 1) 
                emu.frameadvance()
            end
            


            comm.socketServerScreenShot()
            
            
            termination = calulate_termination(step)
            local reward = calculate_reward()
        
            local msg = string.format("%d %d", bool_to_number(termination), reward)
            
            comm.socketServerSend(msg)

            -- draw reward on screen
            gui.cleartext()
            if reward > 0 then 
                gui.drawText(115, 50, tostring(reward), "#00FF00")
            elseif reward < 0 then
                gui.drawText(115, 50, tostring(reward), "#FF0000")
            else
                gui.drawText(115, 50, tostring(reward), "#FFFFFF")
            end

        
            if termination then
                savestate.load("../savestates/start.State")
            end
        
            local lap = mainmemory.readbyte(0x0010C1)
            
            if lap == 133 then
                savestate.load("../savestates/start.State")
                
                -- end the loop and wait for frame done, doesn't send true to model
                termination = true
            end
        
            step = step + 1
            -- Frame done
            comm.socketServerResponse()
        end
    end
    
end

run()