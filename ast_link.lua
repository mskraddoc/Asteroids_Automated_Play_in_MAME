-- ast_link.lua
-- Lock-step link between MAME (Asteroids) and a Python server (ast_server.py).
-- Each emulated frame: send "FRAME <n> <nram> <projN>\n" + RAM bytes (zero page,
-- then object page 0x200), block until "ACK <n> <inputs> <obj>...\n" arrives,
-- press the controls Python chose, then overlay boxes for the objects.
-- <inputs> is letters L R T F H C S (left right thrust fire hyperspace coin start)
-- or "-". Each <obj> is "K,slot,x,y" or, while projection is on,
-- "K,slot,x,y,px,py" where px,py is the position Python projects projN frames
-- ahead; slot is the object's RAM slot (0-34). "A,x,y" is the
-- autopilot's aim point and "MODE:<text>" its current mode.
--
-- Keys: V toggles the projection display, ] / [ increase / decrease projN,
-- I toggles the autopilot (off = you play), B toggles the object boxes and aim
-- line, O toggles the object list.
--
-- Run (from mame/):
--   ./mame asteroid -rompath ../roms -window -autoboot_script ../ast_link.lua

local HOST, PORT = "127.0.0.1", 9999
local TIMEOUT_SECS = 2.0   -- give up waiting for Python after this long
local STATUS_EVERY = 60    -- print a status line every N frames

-- Projection display
local KEY_TOGGLE, KEY_INC, KEY_DEC = "KEYCODE_V", "KEYCODE_CLOSEBRACE", "KEYCODE_OPENBRACE"
local KEY_AI = "KEYCODE_I"
local KEY_BOXES = "KEYCODE_B"
local KEY_LIST = "KEYCODE_O"
local PROJ_STEP, PROJ_MIN, PROJ_MAX = 5, 5, 300
local proj_on = false
local proj_n = 30
local ai_on = true
local boxes_on = true
local list_on = false
local LIST_SCALE = 1.5   -- small-font size: visible-area pixels per glyph grid unit
-- Object overlay: half-width of box in game units (x 0-8191, y 0-6143), color, label
local STYLE = {
	L = { half = 290, color = 0xff00ff00, label = "L" },  -- large rock
	M = { half = 150, color = 0xff00ff00, label = "M" },  -- medium rock
	S = { half =  80, color = 0xff00ff00, label = "S" },  -- small rock
	H = { half = 100, color = 0xffa0a0a0 },               -- ship
	R = { half =  40, color = 0xffff2020 },               -- saucer bullet
	B = { half =  40, color = 0xff3080ff },               -- player bullet
	U = { half = 230, color = 0xffff9000 },               -- large saucer
	u = { half = 120, color = 0xffff9000 },               -- small saucer
}

-- Same hue at half brightness, for projected boxes.
local function dim(color)
	return (color & 0xff000000) | ((color >> 1) & 0x007f7f7f)
end

local machine = manager.machine
local screen = machine.screens[":screen"]
local progspace = machine.devices[":maincpu"].spaces["program"]
local in0 = machine.ioport.ports[":IN0"]
local in1 = machine.ioport.ports[":IN1"]
local input = machine.input
local keys = {
	toggle = input:code_from_token(KEY_TOGGLE),
	inc = input:code_from_token(KEY_INC),
	dec = input:code_from_token(KEY_DEC),
	ai = input:code_from_token(KEY_AI),
	boxes = input:code_from_token(KEY_BOXES),
	list = input:code_from_token(KEY_LIST),
}

-- Input letter from Python -> port field
local CONTROLS = {
	L = in1:field(0x80),  -- rotate left
	R = in1:field(0x40),  -- rotate right
	T = in1:field(0x20),  -- thrust
	F = in0:field(0x10),  -- fire
	H = in0:field(0x08),  -- hyperspace
	C = in1:field(0x01),  -- coin 1
	S = in1:field(0x08),  -- 1 player start
}

local function apply_inputs(letters)
	for letter, field in pairs(CONTROLS) do
		if letters:find(letter, 1, true) then
			field:set_value(1)
		else
			field:clear_value()
		end
	end
end
local key_was_down = {}

-- "rw" connects as a client; "rwc" would make MAME bind/listen on the port instead.
local sock = emu.file("rw")
local err = sock:open("socket." .. HOST .. ":" .. PORT)
if err then
	print("[ast_link] could not connect to " .. HOST .. ":" .. PORT .. " (" .. tostring(err) .. ") - is ast_server.py running?")
	return
end
print("[ast_link] connected to " .. HOST .. ":" .. PORT)

local rxbuf = ""
local frame = 0
local linked = true
local latest_objects = {}
local ai_mode = ""

-- Read one '\n'-terminated line. socket:read never blocks, so poll until a line
-- is complete or the timeout expires. Returns the line (without '\n') or nil.
local function read_line(timeout)
	local deadline = os.clock() + timeout
	while true do
		local nl = rxbuf:find("\n", 1, true)
		if nl then
			local line = rxbuf:sub(1, nl - 1)
			rxbuf = rxbuf:sub(nl + 1)
			return line
		end
		if os.clock() > deadline then
			return nil
		end
		local chunk = sock:read(4096)
		if chunk and #chunk > 0 then
			rxbuf = rxbuf .. chunk
		end
	end
end

-- True only on the frame a key goes down, so holding it doesn't repeat.
local function key_pressed(name)
	local down = input:code_pressed(keys[name])
	local pressed = down and not key_was_down[name]
	key_was_down[name] = down
	return pressed
end

local function poll_keys()
	if key_pressed("toggle") then
		proj_on = not proj_on
	end
	if key_pressed("inc") then
		proj_n = math.min(proj_n + PROJ_STEP, PROJ_MAX)
	end
	if key_pressed("dec") then
		proj_n = math.max(proj_n - PROJ_STEP, PROJ_MIN)
	end
	if key_pressed("boxes") then
		boxes_on = not boxes_on
	end
	if key_pressed("list") then
		list_on = not list_on
	end
	if key_pressed("ai") then
		ai_on = not ai_on
		if not ai_on then
			apply_inputs("")  -- release everything so the keyboard takes over
		end
	end
end

-- Parse "K,x,y[,px,py]" tokens into { {kind, x, y, px, py}, ... }
local function parse_objects(rest)
	local objs = {}
	for tok in rest:gmatch("%S+") do
		local o = {}
		for field in tok:gmatch("[^,]+") do
			o[#o + 1] = #o == 0 and field or tonumber(field)
		end
		objs[#objs + 1] = o
	end
	return objs
end

local function on_frame()
	if not linked then
		return
	end
	frame = frame + 1
	poll_keys()

	local ram = progspace:read_range(0x00, 0xff, 8) .. progspace:read_range(0x200, 0x2ff, 8)
	local n_ahead = proj_on and proj_n or 0
	sock:write("FRAME " .. frame .. " " .. #ram .. " " .. n_ahead .. "\n" .. ram)

	local reply = read_line(TIMEOUT_SECS)
	if not reply then
		print("[ast_link] no reply for frame " .. frame .. " - lock step disabled, emulation continues")
		linked = false
		latest_objects = {}
		return
	end

	local tag, n, inputs, rest = reply:match("^(%S+)%s+(%d+)%s+(%S+)%s*(.*)$")
	if tag ~= "ACK" or tonumber(n) ~= frame then
		print("[ast_link] unexpected reply at frame " .. frame .. ": " .. reply)
		return
	end

	if ai_on then
		apply_inputs(inputs)
	end
	latest_objects = parse_objects(rest)

	if frame % STATUS_EVERY == 0 then
		print("[ast_link] frame " .. frame .. " acked, " .. #latest_objects .. " objects")
	end
end

-- Map game units to screen visible-area pixels (y is up in game, down on screen).
local function to_screen(x, y)
	return x / 8192 * screen.width, (1 - y / 6144) * screen.height
end

local function draw_obj_box(x, y, st, color)
	local x1, y1 = to_screen(x - st.half, y + st.half)
	local x2, y2 = to_screen(x + st.half, y - st.half)
	screen:draw_box(x1, y1, x2, y2, color, 0x00000000)
	return x2, y1
end

-- Small stroke font: each glyph is line segments on a 4x6 grid (y down), written
-- as 4-digit groups "x1 y1 x2 y2". MAME's draw_text has no size option.
local GLYPHS = {
	["0"] = "0040 4046 4606 0600 4006", ["1"] = "2026 1120 1636",
	["2"] = "0040 4043 4303 0306 0646", ["3"] = "0040 4046 4606 1343",
	["4"] = "0003 0343 4046",           ["5"] = "4000 0003 0343 4346 4606",
	["6"] = "4000 0006 0646 4643 4303", ["7"] = "0040 4026",
	["8"] = "0040 4046 4606 0600 0343", ["9"] = "0003 0040 4046 0343 0646",
	A = "0620 2046 1434",               B = "0006 0030 3033 0343 4346 4606",
	C = "4000 0006 0646",               E = "4000 0006 0646 0333",
	H = "0006 4046 0343",               I = "1030 2026 1636",
	K = "0006 0340 0346",               L = "0006 0646",
	M = "0600 0022 2240 4046",          O = "0040 4046 4606 0600",
	P = "0600 0040 4043 4303",          R = "0600 0040 4043 4303 2346",
	S = "4130 3010 1001 0102 0213 1333 3344 4445 4536 3616 1605",
	T = "0040 2026",
	U = "0006 0646 4046",               X = "0046 4006",
	Y = "0023 4023 2326",               ["-"] = "1333",
	[","] = "2516",                     ["#"] = "1115 3135 0232 0434",
}

-- Draw text in the small font with its top-left corner at (x, y) screen pixels.
local function draw_small(x, y, text, color)
	local s = LIST_SCALE
	for i = 1, #text do
		local g = GLYPHS[text:sub(i, i)]
		if g then
			local cx = x + (i - 1) * 6 * s
			for x1, y1, x2, y2 in g:gmatch("(%d)(%d)(%d)(%d)") do
				screen:draw_line(cx + x1 * s, y + y1 * s, cx + x2 * s, y + y2 * s, color)
			end
		end
	end
end

local LIST_INFO = {  -- kind -> type name, score value
	L = { "ROCK-L", "20" },   M = { "ROCK-M", "50" },   S = { "ROCK-S", "100" },
	U = { "SAUCER-L", "200" }, u = { "SAUCER-S", "1000" }, H = { "SHIP", "-" },
	B = { "SHOT-P", "-" },    R = { "SHOT-S", "-" },
}

local function draw_object_list(objs)
	local row_h = 9 * LIST_SCALE
	local x, y = 4, 60
	draw_small(x, y, string.format("%2s %-8s %4s %4s %4s", "#", "TYPE", "X", "Y", "PTS"), 0xffffff00)
	table.sort(objs, function(a, b) return a[2] < b[2] end)
	for _, o in ipairs(objs) do
		y = y + row_h
		local info = LIST_INFO[o[1]]
		draw_small(x, y, string.format("%2d %-8s %4d %4d %4s", o[2], info[1], o[3], o[4], info[2]),
			STYLE[o[1]].color)
	end
end

local function draw_overlay()
	local ship, aim
	local listed = {}
	for _, o in ipairs(latest_objects) do
		local kind, x, y, px, py = o[1], o[3], o[4], o[5], o[6]
		local st = STYLE[kind]
		if kind == "H" then
			ship = o
		elseif kind == "A" then
			aim = o
		elseif kind:sub(1, 5) == "MODE:" then
			ai_mode = kind:sub(6)
		end
		if st then
			listed[#listed + 1] = o
			if boxes_on then
				local lx, ly = draw_obj_box(x, y, st, st.color)
				if st.label then
					screen:draw_text(lx + 2, ly, st.label, st.color, 0x00000000)
				end
			end
			if proj_on and px then
				local c = dim(st.color)
				draw_obj_box(px, py, st, c)
				-- Skip the line when the projection wraps around a screen edge.
				if math.abs(px - x) < 4096 and math.abs(py - y) < 3072 then
					local sx, sy = to_screen(x, y)
					local ex, ey = to_screen(px, py)
					screen:draw_line(sx, sy, ex, ey, c)
				end
			end
		end
	end
	-- Autopilot aim line, ship to predicted intercept (skipped across screen edges).
	-- Aim tokens are "A,x,y"; object tokens carry the slot first, so ship x,y is [3],[4].
	if boxes_on and ai_on and ship and aim
			and math.abs(aim[2] - ship[3]) < 4096 and math.abs(aim[3] - ship[4]) < 3072 then
		local sx, sy = to_screen(ship[3], ship[4])
		local ex, ey = to_screen(aim[2], aim[3])
		screen:draw_line(sx, sy, ex, ey, 0xffffff00)
	end
	if proj_on then
		screen:draw_text(4, 4, "PROJ +" .. proj_n .. " frames", 0xffffff00, 0x00000000)
	end
	screen:draw_text(4, 30, ai_on and ("AI " .. ai_mode) or "AI OFF", 0xffffff00, 0x00000000)
	if list_on then
		draw_object_list(listed)
	end
end

-- Keep the subscription referenced so it is not garbage collected.
ast_link_subscription = emu.add_machine_frame_notifier(on_frame)
emu.register_frame_done(draw_overlay)
