local buf = ""
local c1, c2, c3, c4 = 0, 0, -100, 0

local function init()
  setSerialBaudrate(115200)
end

local function run(event)
  local str = serialRead()
  if str and #str > 0 then
    buf = buf .. str
    for line in string.gmatch(buf, "([^\n]+)\n") do
      local v1, v2, v3, v4 = string.match(line, "(%d+),(%d+),(%d+),(%d+)")
      if v1 and v2 and v3 and v4 then
        -- Convert 1000..2000us to -100..+100 (GVAR range)
        c1 = math.floor((tonumber(v1) - 1500) / 5)
        c2 = math.floor((tonumber(v2) - 1500) / 5)
        c3 = math.floor((tonumber(v3) - 1500) / 5)
        c4 = math.floor((tonumber(v4) - 1500) / 5)
        model.setGlobalVariable(0, 0, c1) -- GV1 (Roll)
        model.setGlobalVariable(1, 0, c2) -- GV2 (Pitch)
        model.setGlobalVariable(2, 0, c3) -- GV3 (Throttle)
        model.setGlobalVariable(3, 0, c4) -- GV4 (Yaw)
      end
    end
    buf = string.match(buf, "[^\n]*$") or ""
  end

  lcd.clear()
  lcd.drawText(2, 2, "PC Serial Control Active", INVERS)
  lcd.drawText(2, 16, string.format("R:%d P:%d", c1, c2))
  lcd.drawText(2, 28, string.format("T:%d Y:%d", c3, c4))
  lcd.drawText(2, 45, "Hold RTN to exit")
  return 0
end

return { init=init, run=run }