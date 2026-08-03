# 五子棋训练助手 / Gomoku Training Assistant

这是一个在 Windows 上运行的本地五子棋训练工具。它读取你选择的游戏窗口或截图，识别 `15 x 15` 棋盘上的黑白棋，调用本地 Rapfi 引擎，并把建议落点显示在助手的棋盘上。

This is a local Windows Gomoku training tool. It reads a selected game window or screenshot, recognizes black and white stones on a `15 x 15` board, calls the local Rapfi engine, and displays suggested moves on the assistant board.

## 当前状态 / Current Status

当前项目是**可运行的 MVP**，已经具备：手工摆局、截图四角校准、Windows 窗口观察、局面合法性检查、Rapfi 限时分析、棋子手数编号和本地复盘记录。

The project is a **runnable MVP**. It includes manual board editing, four-corner screenshot calibration, Windows window observation, legal-state checks, time-bounded Rapfi analysis, move numbering, and local replay logs.

它还不是“打开任意五子棋页面就完全自动识别”的最终产品。第一版只针对一个经过校准的微信、浏览器或桌面客户端棋盘皮肤；换窗口大小、缩放比例或主题后，通常需要重新校准。

It is not yet a zero-configuration recognizer for every Gomoku page. The first version targets one calibrated WeChat, browser, or desktop-client board skin. Resizing, zooming, or changing the theme usually requires recalibration.

规则固定为：`15 x 15`、黑棋先手、无禁手、连成五颗或更多即获胜。

Rules are fixed to `15 x 15`, black first, no forbidden moves, and five or more in a row wins.

该程序仅用于本地训练、复盘或双方知情的测试环境。它不会自动点击其他程序，也不做隐藏或反检测。

The application is for local training, replay, or consented testing. It does not click other applications, conceal itself, or implement anti-detection behavior.

## 最快开始 / Quick Start

当前工作区已经准备好依赖和 Rapfi 引擎时，可直接运行：

If this local workspace already has its dependencies and Rapfi engine prepared,
start it with:

```powershell
.\.venv\Scripts\python -m gomoku_assistant
```

程序打开后按以下顺序操作：

After the application opens, follow these steps:

1. 打开微信或浏览器中的五子棋窗口，并确保整个棋盘可见。
   Open the target WeChat or browser game window and keep the whole board visible.
2. 点击 `Refresh windows`，在 `Target window` 中选择目标窗口。
   Click `Refresh windows`, then select the target under `Target window`.
3. 点击 `Capture frame`，确认右下角预览确实是目标窗口。
   Click `Capture frame` and confirm the lower-right preview is correct.
4. 点击 `Calibrate`，依次点击棋盘的左上、右上、右下、左下四个**交叉点**。
   Click `Calibrate`, then select the top-left, top-right, bottom-right, and bottom-left board **intersections**, in that order.
5. 在 `My color` 中选择 Black、White 或 Analyze both colors，再点击 `Start observing`。Rapfi 会先启动并预热，之后程序会等待三帧一致的画面，并检查新增棋子是否符合黑白轮次；人机快速回应导致一次新增多颗时会自动追赶。
   Choose Black, White, or Analyze both colors under `My color`, then click
   `Start observing`. Rapfi starts and warms up first; then the app waits for
   three matching frames, verifies turn order, and catches up when multiple
   moves appear between samples.
6. 识别到新局面后，助手中央棋盘显示通过战术安全检查的建议。通常最多三个；若只有一个点能挡住对方的立即胜利，则只显示该点；无单点可挡时不显示建议，并以酒红色叉号标出对方的胜点。
   After a new position is recognized, the assistant board shows tactically safe suggestions. It normally shows up to three; when exactly one point prevents an immediate loss, it shows only that point. When no single move can defend, it shows no suggestion and marks the opponent's winning points with burgundy crosses.

如果只想先测试棋力，不需要窗口识别：直接在左侧棋盘点击摆子，然后点击 `Analyze position`。

To test the engine without screen recognition, place stones on the left board and click `Analyze position`.

## 界面说明 / Interface Guide

- `Refresh windows`: 重新扫描可以读取的 Windows 窗口。  
  Rescan available Windows application windows.
- `F5`: 与 `Refresh windows` 相同，只用于发现新窗口或恢复关闭的目标，不需要每回合按。
  Same as `Refresh windows`; use it only to discover a new window or recover a
  closed target, not after every move.
- `Capture frame`: 抓取一次选中窗口，用于确认画面或校准。  
  Capture one frame for confirmation or calibration.
- `Open screenshot`: 导入本地截图，适合先调试棋盘识别。  
  Import a local screenshot for recognition debugging.
- `Calibrate`: 保存四个角点构成的棋盘坐标映射，文件位于 `profiles/default.json`。  
  Save the four-corner board mapping in `profiles/default.json`.
- `Manual correction`: 选择 `Place black`、`Place white` 或 `Erase` 后点击助手棋盘交叉点，修正后的局面会立即用于分析。`Off` 可防止误点，`Auto move` 适合离线手工摆局。
  Choose `Place black`, `Place white`, or `Erase`, then click an assistant-board intersection. The corrected position is used for analysis immediately. `Off` prevents accidental edits; `Auto move` is for offline manual setup.
- `Undo correction` / `Clear corrections`: 撤销最近一次修正，或放弃本局全部人工覆盖并重新从稳定画面同步。青色圈表示人工修正，黄色虚线框表示视觉低置信度格。
  Undo the latest correction, or discard all per-game overrides and resynchronize from stable frames. Cyan circles mark manual corrections; yellow dashed squares mark low-confidence visual cells.
- `My color`: 每局开始前选择自己执黑、执白或双方分析。棋盘始终同步双方落子，但只有轮到你的棋色时才显示推荐。
  Choose black, white, or both before each game. The board always syncs both
  sides, but suggestions appear only when it is your selected color's turn.
- `Black search` / `White search`: 黑白棋独立的 Rapfi 搜索时长。默认黑棋 `8 s`、白棋 `15 s`；界面和引擎都会拒绝超过 `15 s` 的配置。
  Independent Rapfi search durations. Defaults are `8 s` for black and `15 s` for white; both the UI and engine reject values above `15 s`.
- `Rapfi threads` / `Rapfi hash`: 默认使用 `8` 个线程和 `512 MB` 搜索哈希。更改线程或哈希会在下一次分析前重建 Rapfi；黑白时间切换不会重启引擎。
  Defaults are `8` threads and a `512 MB` search hash. Changing either takes effect by rebuilding Rapfi before the next analysis; switching black/white time does not restart it.
- `Select Rapfi.exe`: 仅在更换引擎文件时使用。  
  Use this only when changing the engine executable.

## 技术架构 / Architecture

数据按以下路径流动：

`游戏窗口或截图 -> 原始棋盘识别 -> 人工修正层 -> 合法局面确认 -> Rapfi 分析 -> 助手棋盘提示`

Data flows through:

`game window or screenshot -> raw board recognition -> manual correction layer -> legal-state confirmation -> Rapfi analysis -> assistant board hint`

1. **采集层 / Capture**
   Qt 从你选择的 Windows 窗口读取画面，也可以加载本地截图。程序只读取像素，不向目标窗口发送鼠标或键盘操作。

   Qt reads the selected Windows window or a local image. The app reads pixels only and never sends mouse or keyboard input to the target.

2. **视觉层 / Vision**
   你选择四个棋盘角点后，OpenCV 会将棋盘拉正为标准正方形，再检查 225 个交叉点。它结合绝对颜色、与周围棋盘的相对明暗、上一手标记、边缘半棋子和小范围中心偏移，输出空、黑、白三类证据。

   After you select four board corners, OpenCV warps the board into a standard square and evaluates all 225 intersections. It combines absolute color, relative board contrast, last-move markers, edge half-stones, and small center offsets into empty, black, and white evidence.

3. **状态保护层 / State Validation**
   动画、上一手标记或弹窗可能会干扰识别。因此程序融合连续三帧的逐格证据，并验证棋子数量、轮次和合法的多步追赶。只有一个低置信度格能形成唯一合法局面时才会自动修复；其他歧义会标出并等待人工修正。菜单、广告或没有清晰网格的页面会暂停提示，而不是猜测局面。

   Animations, last-move markers, and dialogs can confuse recognition. The app
   fuses per-cell evidence across three frames and checks stone counts, turn
   order, and legal multi-move catch-up. It repairs one low-confidence cell
   only when that produces the unique legal state; otherwise it marks the
   ambiguity for manual correction. Menus, ads, and frames without a clear grid
   pause observation instead of guessing.

4. **棋力层 / Engine**
   Rapfi 在本地进程中运行，根据当前局面搜索最佳走法。随附版本稳定输出最佳一手；程序会先处理立即胜负和唯一防守，再将 Rapfi 与本地候选进行即时杀和双威胁安全检查。只有真正安全的点才会显示，最多三个，不会为了凑数补充危险落点。

   Rapfi runs as a local process and searches the best move. The bundled version reliably returns one best move; the app first resolves immediate wins, losses, and mandatory blocks, then checks Rapfi and local alternatives for immediate losses and double threats. Only genuinely safe moves are shown, up to three, with no unsafe padding.

5. **显示与记录层 / Display and Logging**
   助手棋盘显示当前局面、可靠的棋子手数和候选，并标出最后确认的一手。人工修正会在本局持续覆盖自动识别，直到新局、重新开始观察、切换窗口、重新标定或主动清除。关闭程序时会把逐手记录、纠错事件、分析参数、实际搜索时间、深度、战术模式、危险点和被拒绝的候选保存到 `sessions/`，可用于复盘。

   The assistant board shows the position, reliable move numbers, the latest confirmed move, and suggestions. Manual corrections override visual recognition for the current game until a new game, a new observation run, a target switch, recalibration, or an explicit clear. On close, it saves moves, correction events, analysis parameters, elapsed time, depth, and terminal results to `sessions/` for replay.

## 搜索时限 / Search Limits

对局计时是硬约束，因此助手不允许把引擎时间拉长到无限。引擎每步最多搜索 `15 s`，从助手确认棋盘到返回或丢弃结果的总期限为 `17 s`，给窗口识别、显示和实际点击保留余量。首次 `Start observing` 会在开始同步棋局前预热 Rapfi，不占用某一步的思考时间。

The game clock is a hard constraint, so the assistant never uses an unbounded search. Rapfi receives at most `15 s` per move, and the assistant returns or discards the result within a `17 s` total deadline after it confirms a board. This leaves time for recognition, display, and the actual click. The first `Start observing` warms Rapfi before board synchronization begins, outside a move's search budget.

Rapfi 的深度由引擎在给定时间内自动迭代加深。助手会记录实际达到的深度，但不修改 Rapfi 的评估模型、权重或源代码；日志用于复盘和比较参数，而不会自动“训练”或覆盖引擎推荐。

Rapfi chooses its own iterative-deepening depth inside the supplied time. The assistant records the reached depth but does not change Rapfi's evaluator, weights, or source code. Logs support replay and parameter comparison only; they do not train or override the engine.

要将已有日志和新参数进行离线对比，可运行以下命令。它默认抽取 20 个不同局面，黑棋按 8 秒、白棋按 15 秒复算，并在 `benchmarks/` 输出报告；每个局面同样不会超过 15 秒引擎时间。

To compare existing logs against the new settings offline, run the command below. It samples 20 unique positions by default, recalculates black at 8 seconds and white at 15 seconds, and writes a report under `benchmarks/`; each position remains capped at 15 engine seconds.

```powershell
.\.venv\Scripts\python -m gomoku_assistant.benchmark --rapfi vendor\rapfi\Rapfi.exe
```

## Rapfi 引擎 / Rapfi Engine

本机工作区可以将 Rapfi Windows 引擎包放在 `vendor/rapfi/`。为避免将大二进制和
评估权重同步到 GitHub，远端仓库只保留该目录的说明文件，不包含 `Rapfi.exe` 和
`.bin.lz4` 权重。

The local workspace can keep the Rapfi Windows package in `vendor/rapfi/`.
To avoid syncing large binaries and model weights to GitHub, the remote
repository keeps only this directory's instructions, not `Rapfi.exe` or
`.bin.lz4` weights.

- `Rapfi.exe` 是当前默认使用的 AVX2 版本。  
  `Rapfi.exe` is the default AVX2 build.
- `config.toml` 和 `.bin.lz4` 是匹配的配置与评估权重。  
  `config.toml` and `.bin.lz4` are matching configuration and evaluation weights.
- `COPYING.txt` 是 GPLv3 许可证。  
  `COPYING.txt` contains the GPLv3 license.

从 GitHub 克隆后，如需强引擎分析，请下载官方 Rapfi Windows 发行包并解压到
`vendor/rapfi/`，或在程序中通过 `Select Rapfi.exe` 选择现有引擎文件。没有 Rapfi
时，程序仍可运行，但只会使用本地战术兜底。

After cloning from GitHub, download an official Rapfi Windows release into
`vendor/rapfi/`, or select an existing engine through `Select Rapfi.exe` for
strong analysis. Without Rapfi, the app still runs with the local tactical
fallback only.

如果项目复制到不支持 AVX2 的旧电脑，请通过 `Select Rapfi.exe` 选择 `pbrain-rapfi-windows-sse.exe`。

If the project is copied to a computer without AVX2 support, use `Select Rapfi.exe` to choose `pbrain-rapfi-windows-sse.exe`.

## 识别不准时 / When Recognition Is Inaccurate

1. 确认四条边和四个角点都没有被聊天栏、弹窗或广告遮挡。  
   Make sure all board edges and corners are not covered by chat, dialogs, or ads.
2. 确认校准时点击的是棋盘**交叉点**，不是外框。  
   Confirm that calibration uses board **intersections**, not the outer frame.
3. 在窗口缩放、主题切换或 DPI 改变后重新校准。  
   Recalibrate after window resizing, theme changes, or DPI changes.
4. 每个小程序首次使用时重新标定。程序会按窗口标题和采集尺寸保存独立配置，之后切回同一窗口会自动加载。
   Calibrate each mini-program once. The app saves one profile per window title
   and capture size, then loads it automatically when that window is selected.
5. 优先收集空棋盘、早中晚盘、上一手标记、胜负动画和不同窗口尺寸的截图。这些截图用于调节视觉阈值和补充回归测试，不用于训练新的棋力模型。
   Collect screenshots of empty, early, middle, and late boards, last-move markers, win animations, and different window sizes. They tune visual thresholds and regression tests; they do not train a new game-strength model.
6. 如果自动棋盘少子或多子，先观察黄色虚线格；在 `Manual correction` 选择黑、白或擦除并点击该格。修正会持续到本局结束，随后按新的中央棋盘自动计算建议。
   If the automatic board misses or adds a stone, inspect yellow dashed cells, choose black, white, or erase under `Manual correction`, and click that point. The correction persists for the game and suggestions are recalculated from the corrected assistant board.

## 开发命令 / Development Commands

重新安装依赖：

Reinstall dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

运行测试：

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```
