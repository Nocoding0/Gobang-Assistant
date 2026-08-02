# 五子棋训练助手 / Gomoku Training Assistant

这是一个在 Windows 上运行的本地五子棋训练工具。它读取你选择的游戏窗口或截图，识别 `15 x 15` 棋盘上的黑白棋，调用本地 Rapfi 引擎，并把建议落点显示在应用棋盘和可选透明提示层上。

This is a local Windows Gomoku training tool. It reads a selected game window or screenshot, recognizes black and white stones on a `15 x 15` board, calls the local Rapfi engine, and displays suggested moves on the app board and an optional transparent overlay.

## 当前状态 / Current Status

当前项目是**可运行的 MVP**，已经具备：手工摆局、截图四角校准、Windows 窗口观察、局面合法性检查、Rapfi 引擎分析、透明提示层和本地复盘记录。

The project is a **runnable MVP**. It includes manual board editing, four-corner screenshot calibration, Windows window observation, legal-state checks, Rapfi analysis, a transparent overlay, and local replay logs.

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
5. 点击 `Start observing`。程序会等待三帧一致的画面，并检查是否只新增了一颗、且黑白轮次正确的棋子。
   Click `Start observing`. The app waits for three matching frames and checks that only one correctly colored stone was added.
6. 识别到新局面后，红色 `1` 是 Rapfi 的首选。其余两个候选由本地战术算法补充；勾选 `Show transparent overlay` 后，标记会覆盖在目标窗口上。
   After a new position is recognized, red `1` is Rapfi's preferred move. The other two moves are local tactical alternatives. Enable `Show transparent overlay` to draw markers over the target window.

如果只想先测试棋力，不需要窗口识别：直接在左侧棋盘点击摆子，然后点击 `Analyze position`。

To test the engine without screen recognition, place stones on the left board and click `Analyze position`.

## 界面说明 / Interface Guide

- `Refresh windows`: 重新扫描可以读取的 Windows 窗口。  
  Rescan available Windows application windows.
- `Capture frame`: 抓取一次选中窗口，用于确认画面或校准。  
  Capture one frame for confirmation or calibration.
- `Open screenshot`: 导入本地截图，适合先调试棋盘识别。  
  Import a local screenshot for recognition debugging.
- `Calibrate`: 保存四个角点构成的棋盘坐标映射，文件位于 `profiles/default.json`。  
  Save the four-corner board mapping in `profiles/default.json`.
- `Edit board`: 设置手工摆子的方式；`Auto move` 会按照黑白正常轮次落子。  
  Set manual editing behavior; `Auto move` follows the normal black/white turn order.
- `Analyze turns`: 观察模式下可选择只在黑棋或白棋回合分析。  
  In observation mode, choose analysis only on black turns or white turns.
- `Rapfi search`: 每个局面的搜索时长，默认 `1000 ms`。  
  Search duration per position; the default is `1000 ms`.
- `Select Rapfi.exe`: 仅在更换引擎文件时使用。  
  Use this only when changing the engine executable.

## 技术架构 / Architecture

数据按以下路径流动：

`游戏窗口或截图 -> 棋盘识别 -> 合法局面确认 -> Rapfi 分析 -> 棋盘和透明层提示`

Data flows through:

`game window or screenshot -> board recognition -> legal-state confirmation -> Rapfi analysis -> board and overlay hint`

1. **采集层 / Capture**
   Qt 从你选择的 Windows 窗口读取画面，也可以加载本地截图。程序只读取像素，不向目标窗口发送鼠标或键盘操作。

   Qt reads the selected Windows window or a local image. The app reads pixels only and never sends mouse or keyboard input to the target.

2. **视觉层 / Vision**
   你选择四个棋盘角点后，OpenCV 会将棋盘拉正为标准正方形，再检查 225 个交叉点，判断它们为空、黑棋或白棋。

   After you select four board corners, OpenCV warps the board into a standard square and classifies all 225 intersections as empty, black, or white.

3. **状态保护层 / State Validation**
   动画、上一手标记或弹窗可能会干扰识别。因此程序要求连续三帧一致，并验证棋子数量、轮次和“仅新增一颗棋子”。不满足时会暂停提示，而不是猜测局面。

   Animations, last-move markers, and dialogs can confuse recognition. The app requires three stable frames and checks stone counts, turn order, and a single new stone. It pauses instead of guessing.

4. **棋力层 / Engine**
   Rapfi 在本地进程中运行，根据当前局面搜索最佳走法。随附版本稳定输出最佳一手；程序再用本地战术算法补足第二和第三候选，所以不会把它们伪装成 Rapfi 的精确评分。

   Rapfi runs as a local process and searches the best move. The bundled version reliably returns one best move; the app uses a local tactical algorithm for second and third alternatives, without presenting them as exact Rapfi scores.

5. **显示与记录层 / Display and Logging**
   左侧棋盘显示当前局面和候选。透明提示层可映射回目标窗口；关闭程序时会把分析记录保存到 `sessions/`，可用于复盘。

   The left board shows the current position and suggestions. The transparent overlay maps them back to the target window. Analysis records are saved to `sessions/` when the app closes.

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
4. 优先收集空棋盘、早中晚盘、上一手标记、胜负动画和不同窗口尺寸的截图。这些截图用于调节视觉阈值和补充回归测试，不用于训练新的棋力模型。  
   Collect screenshots of empty, early, middle, and late boards, last-move markers, win animations, and different window sizes. They tune visual thresholds and regression tests; they do not train a new game-strength model.

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
