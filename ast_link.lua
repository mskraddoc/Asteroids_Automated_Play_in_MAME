-- ast_link.lua
-- Lock-step link between MAME (Asteroids) and a Python server (ast_server.py).
-- Each emulated frame: send "FRAME <n> <nram> <projN> <style_step>\n" + RAM bytes (zero page,
-- then object page 0x200), block until "ACK <n> <inputs> <obj>...\n" arrives,
-- press the controls Python chose, then overlay boxes for the objects.
-- <inputs> is letters L R T F H C S (left right thrust fire hyperspace coin start)
-- or "-". Each <obj> is "K,slot,x,y" or, while projection is on,
-- "K,slot,x,y,px,py" where px,py is the position Python projects projN frames
-- ahead; slot is the object's RAM slot (0-34). "A,x,y" is the
-- autopilot's aim point, "MODE:<text>" its current mode and
-- "G,score,ships,level,deaths,hyper,coll,shot,kami,jumps" the game totals for the
-- score overlay (ships lost by cause, then hyperspace jumps). "SEED,lo,hi" (sent at game start) is
-- written into the game's random-number generator at zero page 0x5F/0x60.
-- "PET,slot" marks the shadow style's spared boulder. <style_step> is 1 on the
-- frame M is pressed (Python switches to the next play style), else 0.
--
-- Keys: V toggles the projection display, ] / [ increase / decrease projN,
-- I toggles the autopilot (off = you play), B toggles the object boxes and aim
-- line, O toggles the object list, A toggles the AI state text, S toggles the
-- score overlay, . / , speed emulation up / down (0.25x-8x, then MAX =
-- unthrottled, which shows the measured speed), = / - raise / lower volume,
-- M switches to the next autopilot play style.
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
local KEY_AI_TEXT = "KEYCODE_A"
local KEY_SCORE = "KEYCODE_S"
local KEY_FASTER, KEY_SLOWER = "KEYCODE_STOP", "KEYCODE_COMMA"
local KEY_VOL_UP, KEY_VOL_DOWN = "KEYCODE_EQUALS", "KEYCODE_MINUS"
local KEY_STYLE = "KEYCODE_M"
local PROJ_STEP, PROJ_MIN, PROJ_MAX = 5, 5, 300
local proj_on = false
local proj_n = 30
local ai_on = true
local boxes_on = true
local list_on = false
local ai_text_on = true
local score_on = true
local SPEEDS = { 0.25, 0.5, 1, 2, 4, 8, "MAX" }  -- speed multipliers; MAX = unthrottled
local speed_i = 3                          -- 1x, normal speed
local VOL_STEP, VOL_MIN, VOL_MAX = 2, -32, 0  -- MAME attenuation, dB
local NOTICE_SECS = 3                      -- how long the speed / volume notice shows
local MEASURE_SECS = 0.5                   -- real-time window for the measured speed
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
	ai_text = input:code_from_token(KEY_AI_TEXT),
	score = input:code_from_token(KEY_SCORE),
	faster = input:code_from_token(KEY_FASTER),
	slower = input:code_from_token(KEY_SLOWER),
	vol_up = input:code_from_token(KEY_VOL_UP),
	vol_down = input:code_from_token(KEY_VOL_DOWN),
	style = input:code_from_token(KEY_STYLE),
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
local game_stats    -- latest "G" token: { "G", score, ships, level, deaths, 4 causes, jumps }
local notice, notice_until = nil, 0  -- speed / volume message (string or function) and expiry
local measured_speed                 -- emulated seconds per real second, last window
local t_wall0, t_emu0                -- start of the current measuring window
local style_step = 0                 -- 1 asks Python for the next play style

-- Wall-clock seconds; frame counts would stretch or shrink with the speed setting.
local function now()
	return emu.osd_ticks() / emu.osd_ticks_per_second()
end

local function show_notice(text)
	notice, notice_until = text, now() + NOTICE_SECS
end

local function fmt_speed(x)
	return (x == math.floor(x) and string.format("%d", x) or tostring(x)) .. "X"
end

-- Called every frame: emulated time over real time, averaged per MEASURE_SECS window.
local function measure_speed()
	local wall, emu_t = now(), machine.time:as_double()
	if not t_wall0 then
		t_wall0, t_emu0 = wall, emu_t
	elseif wall - t_wall0 >= MEASURE_SECS then
		measured_speed = (emu_t - t_emu0) / (wall - t_wall0)
		t_wall0, t_emu0 = wall, emu_t
	end
end

local function max_speed_notice()
	if measured_speed then
		return string.format("SPEED MAX  (measured %.1fX)", measured_speed)
	end
	return "SPEED MAX  (measuring...)"
end

local function set_speed(i)
	i = math.max(1, math.min(i, #SPEEDS))
	if i ~= speed_i then
		t_wall0, measured_speed = nil, nil  -- start measuring the new setting afresh
	end
	speed_i = i
	local speed = SPEEDS[speed_i]
	if speed == "MAX" then
		machine.video.throttled = false
		show_notice(max_speed_notice)
	else
		machine.video.throttled = true
		machine.video.throttle_rate = speed
		local s = "SPEED " .. fmt_speed(speed)
			.. "  (" .. fmt_speed(SPEEDS[1]) .. " - " .. fmt_speed(SPEEDS[#SPEEDS - 1]) .. ", MAX)"
		if speed_i == 1 then
			s = s .. "  MIN"
		end
		show_notice(s)
	end
end

-- The style is the first word of the AI mode, e.g. "SHADOW 1/2 FOLLOW".
local function style_notice()
	return "Style: " .. (ai_mode:match("^%S+") or "")
end

local function change_volume(step)
	local sound = machine.sound
	sound.attenuation = math.max(VOL_MIN, math.min(sound.attenuation + step, VOL_MAX))
	show_notice("VOL " .. sound.attenuation .. " dB")
end

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
	if key_pressed("ai_text") then
		ai_text_on = not ai_text_on
	end
	if key_pressed("score") then
		score_on = not score_on
	end
	if key_pressed("faster") then
		set_speed(speed_i + 1)
	end
	if key_pressed("slower") then
		set_speed(speed_i - 1)
	end
	if key_pressed("vol_up") then
		change_volume(VOL_STEP)
	end
	if key_pressed("vol_down") then
		change_volume(-VOL_STEP)
	end
	if key_pressed("style") then
		style_step = 1
		show_notice(style_notice)
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
	measure_speed()

	local ram = progspace:read_range(0x00, 0xff, 8) .. progspace:read_range(0x200, 0x2ff, 8)
	local n_ahead = proj_on and proj_n or 0
	sock:write("FRAME " .. frame .. " " .. #ram .. " " .. n_ahead .. " " .. style_step .. "\n" .. ram)
	style_step = 0

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
	for _, o in ipairs(latest_objects) do
		if o[1] == "SEED" then
			progspace:write_u8(0x5f, o[2])
			progspace:write_u8(0x60, o[3])
		end
	end

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
	N = "0006 0046 4640",
	U = "0006 0646 4046",               X = "0046 4006",
	Y = "0023 4023 2326",               Z = "0040 4006 0646",
	["-"] = "1333",
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
	local x, y = 4, 90  -- below the two-line AI text
	draw_small(x, y, string.format("%2s %-8s %4s %4s %4s", "#", "TYPE", "X", "Y", "PTS"), 0xffffff00)
	table.sort(objs, function(a, b) return a[2] < b[2] end)
	for _, o in ipairs(objs) do
		y = y + row_h
		local info = LIST_INFO[o[1]]
		draw_small(x, y, string.format("%2d %-8s %4d %4d %4s", o[2], info[1], o[3], o[4], info[2]),
			STYLE[o[1]].color)
	end
end

-- 1234567 -> "1,234,567"
local function commas(n)
	local s = tostring(math.floor(n))
	s = s:reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", "")
	return s
end

local PET_COLOR = 0xffff00ff

-- Two lines: the play style (first word of the AI mode), then the rest of the mode.
local function ai_text()
	local style, rest = ai_mode:match("^(%S+)%s*(.*)$")
	local state = ai_on and ("AI " .. (rest or "")) or "AI OFF"
	return style and ("Style: " .. style .. "\n" .. state) or state
end

-- Score overlay lines from the latest "G" token.
local function score_text()
	return string.format("SCORE %s\nSHIPS %d\nLEVEL %d\nDEATHS %d",
		commas(game_stats[2]), game_stats[3], game_stats[4], game_stats[5] or 0)
end

-- Small-font lines under DEATHS: ships lost by cause (G token fields 6-9), each
-- with the color its line takes for FLASH_SECS after a loss of that kind.
local CAUSES = {
	{ "HYPER", 0xff3080ff },       -- failed hyperspace jump
	{ "COLLISIONS", 0xffff9000 },  -- hit a rock or saucer
	{ "SHOT", 0xffff2020 },        -- hit by a saucer bullet
	{ "KAMIKAZE", 0xffff00ff },    -- spent over the autopilot's ship cap
}
local FLASH_SECS = 2  -- wall clock, so the same at any emulation speed
local cause_seen = {}   -- count last drawn, per cause
local flash_until = {}  -- wall-clock time each cause's line stops flashing

-- Draw the cause lines right-aligned from y down; returns the y below them.
local function draw_causes(y)
	for i, c in ipairs(CAUSES) do
		local n = game_stats[5 + i] or 0
		if cause_seen[i] and n > cause_seen[i] then
			flash_until[i] = now() + FLASH_SECS
		end
		cause_seen[i] = n
		local text = string.format("%-10s %5d", c[1], n)
		local color = now() < (flash_until[i] or 0) and c[2] or 0xffffff00
		draw_small(screen.width - #text * 6 * LIST_SCALE - 4, y, text, color)
		y = y + 9 * LIST_SCALE
	end
	return y
end

local function draw_overlay()
	local ship, aim, pet_slot
	local listed = {}
	for _, o in ipairs(latest_objects) do
		if o[1] == "PET" then
			pet_slot = o[2]
		end
	end
	for _, o in ipairs(latest_objects) do
		local kind, x, y, px, py = o[1], o[3], o[4], o[5], o[6]
		local st = STYLE[kind]
		local is_pet = st and o[2] == pet_slot
		if kind == "H" then
			ship = o
		elseif kind == "A" then
			aim = o
		elseif kind == "G" then
			game_stats = o
		elseif kind:sub(1, 5) == "MODE:" then
			ai_mode = (kind:sub(6):gsub("_", " "))
		end
		if st then
			listed[#listed + 1] = o
			if boxes_on then
				local color = is_pet and PET_COLOR or st.color
				local lx, ly = draw_obj_box(x, y, st, color)
				local label = is_pet and "PET" or st.label
				if label then
					screen:draw_text(lx + 2, ly, label, color, 0x00000000)
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
	if ai_text_on then
		screen:draw_text(4, 30, ai_text(), 0xffffff00, 0x00000000)
	end
	if score_on and game_stats then
		screen:draw_text("right", 4, score_text(), 0xffffff00, 0x00000000)
		local line_h = manager.ui.line_height * screen.height  -- draw_text's line height
		local y = draw_causes(4 + 4 * line_h + 2)
		screen:draw_text("right", y + 2, "HYPERSPACE " .. (game_stats[10] or 0), 0xffffff00, 0x00000000)
	end
	if notice and now() < notice_until then
		local text = type(notice) == "function" and notice() or notice
		screen:draw_text("center", screen.height / 4, text, 0xffffff00, 0x00000000)
	end
	if list_on then
		draw_object_list(listed)
	end
end

-- Keep the subscription referenced so it is not garbage collected.
ast_link_subscription = emu.add_machine_frame_notifier(on_frame)
emu.register_frame_done(draw_overlay)
