# CoinVision MG400 Control

這個資料夾是目前雙相機硬幣辨識 + MG400 手臂座標輸出的整理包。

系統目的：
- Quality camera / Logitech C270：負責看清楚硬幣邊緣、SAM3 + ellipse 偵測、分類 1 / 5 / 10 / 50 元。
- Gemini2 depth camera：負責深度、桌面座標對應、輸出硬幣在 MG400 座標系的位置。
- MG400：目前是 dry run 下爪測試，還沒有控制真空吸附 DO。

目前操作主程式是：

```bat
RUN_COIN_ROBOT_UI.bat
```

如果在原本開發資料夾操作，也可以直接用：

```bat
C:\Users\user\Desktop\coin_classifier\run_coin_robot_ui.bat
```

## 資料夾內容

- `src/`：主要 Python 程式，已把目前的 JSON 設定也複製一份放在這裡，方便直接執行。
- `run/`：原本開發過程使用的 bat 檔備份。
- `config/`：目前標定與 UI 使用設定。
- `calibration/`：部分標定預覽、除錯圖片與歷史資料。
- `external_core/`：從 `sam3+座標轉換與夾取/core` 複製過來的相機、手臂、轉換模組備份。
- `src/core/`：同一份 core 的可執行副本，package 版程式會優先用這裡。
- `AGENT.md`：給其他 AI / 工程師快速接手用。
- `README.md`：給人看的完整流程。

## 目前硬體設定

- Gemini2：深度相機，彩色串流 1280x720 RGB 30fps。
- Quality camera：Logitech C270，camera index 3，1280x720 30fps。
- 棋盤格：內角點 9x6，單格 26 mm x 26 mm。
- MG400 起始避開相機位置：`X=30, Y=280, Z=150`。
- 目前下爪測試高度：`Z=-156`，桌面模型中的 table Z 是 `-160`。
- 目前手臂目標補償：`X=+13 mm, Y=-18 mm`，寫在 `src/dual_camera_config.json` 的 `robot_target_offset_x_mm / robot_target_offset_y_mm`。
- 自動 Pick 保守範圍：`X=120..380 mm, Y=-250..190 mm`；範圍外會標成 `Check`，不要直接 dry lower。
- 目前手臂速度預設：移動 40%，下降 25%，返回/高位移動 40%。

## 快速使用流程

1. 確認 Gemini、Quality camera、MG400 都接好。
2. 關掉其他會佔用相機的程式，例如舊的 UI、相機預覽、DobotStudio 影像工具。
3. 開啟 `RUN_COIN_ROBOT_UI.bat`。
4. UI 開啟後只顯示 Quality camera 的 ROI 內即時畫面，不顯示辨識 overlay，也不載入 SAM3 或啟動 Gemini 深度辨識。
5. 右上角 `設定` 可以切換操作員/工程師模式、中英文介面、移動速度、下降速度。
6. 相機畫面上方可以切換 `畫質相機 / 深度相機 / 雙相機`。平常最快的是畫質相機 ROI 即時畫面。
7. 操作員模式只保留日常手臂流程；工程師模式會顯示 `設定辨識 ROI`、`重新辨識 / 鎖定座標`、選取硬幣測試、全部測試等進階按鈕。
8. 要測位置時按 `辨識後移到第一顆上方`。系統會先回 camera-clear pose，再自動辨識並移到第一個可取硬幣上方。
9. 要 dry lower 時按 `辨識後下降第一顆到 Z=-156`。系統會在動作前自動辨識並鎖定座標。
10. 如果 MG400 報錯，UI 會跳出錯誤碼、目標座標、可能原因和建議處理。先確認手臂安全，再按 `清除報警並啟用手臂`。
11. 急停請按 UI 的 `急停 / 停用手臂`，它會送 `DisableRobot()` 並停止目前動作程式。這不是實體急停的替代品。

## 重要觀念

畫面上的即時預覽不是最終辨識結果。

目前為了讓 UI 快，平常只做 ROI 內的 Quality camera 即時預覽。真正的硬幣辨識、深度取樣、座標轉換只在開始手臂動作時執行。一般操作介面不提供手動 `Refresh / Lock Vision`、`Set Detection ROI`、相機切換或速度滑桿，避免誤操作。

## 目前座標轉換方式

目前實際用於 MG400 的座標流程是：

1. Quality camera 用 SAM3 + ellipse 找硬幣中心、邊緣和直徑。
2. 用 `quality_to_gemini_homography.json` 把 Quality 畫面座標轉到 Gemini 彩色/深度畫面座標。
3. Gemini 在對應像素附近取深度，輸出硬幣的 Z / 深度資訊。
4. 用 `robot_tabletop_homography.json` 把 Gemini 畫面座標轉成 MG400 的 `X/Y`。
5. `Z` 目前不是由深度直接轉 MG400 世界座標，而是使用桌面固定高度：

```text
robot_table_z_mm = -160.0
dry lower target Z = -156
robot target offset = X +13 mm, Y -18 mm
```

也就是說，目前的手臂座標轉換是「桌面平面 homography」：

```text
Quality pixel -> Gemini pixel -> MG400 tabletop X/Y
```

不是完整 3D 世界座標，也不是每次用 ArUco 求位姿。

## 目前標定精度

Quality camera 內參：
- 檔案：`config/quality_camera_calib.json`
- 影像尺寸：1280x720
- 棋盤格：9x6 inner corners，26 mm
- 使用照片：22 張
- reprojection error：約 `0.2887 px`

Quality 到 Gemini homography：
- 檔案：`src/quality_to_gemini_homography.json`
- 用途：Quality pixel 對應到 Gemini pixel。
- 使用同步拍攝的棋盤格桌面多位置照片。
- 目前使用配對：18 組。
- mean error：約 `0.22 px`
- max error：約 `9.01 px`
- 這不是世界座標，只是兩台相機畫面互相對應。

Gemini 到 MG400 桌面 homography：
- 檔案：`src/robot_tabletop_homography.json`
- 使用 12 個桌面人工點位。
- mean error：約 `0.907 mm`
- max error：約 `2.707 mm`
- 目前是手臂落點準確的主要依據。

Stereo extrinsics：
- 檔案：`config/stereo_extrinsics.json`
- 之前有做雙相機外參，但目前 MG400 落點主要不靠這個，而是靠桌面 homography。

## 什麼時候要重標定

### 只移動 Quality camera

需要重做：

1. Quality camera 內參如果焦距、解析度、鏡頭設定改很多，重跑 `run_quality_calib.bat` 或 `src/capture_quality_calib.py` + `src/calibrate_camera.py`。
2. Quality 到 Gemini 對應一定要重做：拍棋盤格平放桌面、多個位置，跑 `run_quality_to_gemini_calib.bat` 或 `src/calibrate_quality_to_gemini_homography.py`。

不一定需要重做：

- Gemini 到 MG400 桌面 homography，只要 Gemini 沒動、桌子和手臂沒動，通常不用。

### 只移動 Gemini camera

需要重做：

1. Quality 到 Gemini homography。
2. Gemini 到 MG400 桌面 homography。
3. Gemini ROI / 曝光 / 顯示裁切可能也要重新調。

### 桌子、MG400 底座、相機架任何一個移動

需要重做：

1. Gemini 到 MG400 桌面 homography。
2. Quality 到 Gemini homography 建議也重做，尤其兩台相機視角相對桌面改變時。

### 改解析度或 Gemini ROI

如果改的是「顯示裁切 ROI」，不改實際相機原始座標，通常不用重標。

如果改的是相機輸出解析度，例如 1280x720 改成 640x480，既有 pixel 座標標定會失效，需要重標：

1. Quality 內參。
2. Quality 到 Gemini homography。
3. Gemini 到 MG400 桌面 homography。

## 相機內參重標流程

Quality camera 內參：

1. 使用 9x6 內角點棋盤格，單格 26 mm。
2. 棋盤格放在桌面，拍 15 到 25 張以上。
3. 每張角度要不同，包含左上、右上、左下、右下、中央，不要全部只拍正中間。
4. 棋盤格要清楚、不要反光、不要切到邊。
5. 跑 Quality 內參標定程式。
6. 確認 reprojection error 低於約 0.5 px；目前這組約 0.2887 px。
7. 產生或更新 `quality_camera_calib.json`。

Gemini 本身內參目前由 Orbbec pipeline 讀取，程式啟動時會印出 fx / fy / cx / cy。若要做完整 Gemini 內參備份，可以另外拍棋盤格標定，但目前流程主要靠 SDK 內參 + 桌面 homography。

## 兩台相機外參 / 對應重標流程

目前不是做完整雙相機世界外參，而是做 Quality 到 Gemini 的畫面對應。

1. 把 9x6 棋盤格平放在桌面。
2. 移到桌面不同位置拍多組 Gemini + Quality 照片。
3. 棋盤格要完整出現在兩台相機畫面中。
4. 建議至少 15 組，越平均覆蓋桌面越好。
5. 跑 `calibrate_quality_to_gemini_homography.py`。
6. 產生 `quality_to_gemini_homography.json`。
7. 檢查輸出的 debug 圖，確認 Quality 偵測點投到 Gemini 位置合理。

這一步的目的只是讓兩台相機「知道同一顆硬幣是哪一顆」，不是把兩台相機都轉到世界座標。

## 手臂座標重標流程

目前手臂座標使用 `robot_tabletop_homography.json`。

重標時要做的是：建立 Gemini 畫面上的桌面點，對應到 MG400 實際 XY。

基本流程：

1. 讓 MG400 TCP / 指針可以安全地指到桌面點位。
2. 開啟 Gemini 畫面，選桌面上容易對齊的點，例如硬幣中心、棋盤角點、貼紙十字。
3. 對每個點記錄：
   - Gemini image pixel：`u, v`
   - MG400 實際座標：`X, Y`
4. 點位至少 4 點，建議 9 到 16 點。
5. 點位要分散在整個可抓取區域，不要都集中中間。
6. 跑 `calibrate_robot_tabletop_homography.py`。
7. 產生 `robot_tabletop_homography.json`。
8. 看 mean error / max error。現在這組 mean 約 0.91 mm，max 約 2.71 mm。
9. 開 UI 做 hover 或 dry lower 測試。先用高 Z hover，再測低 Z。

目前不一定需要 ArUco。

ArUco 可以用來做更自動的相機位姿估計，但現在實際穩定落點是靠桌面 homography。只要桌子是平面，硬幣都放在桌面上，homography 很直接、好調、也比較容易人工驗證。

## 如果要讓手臂座標更穩

建議優先做這些：

1. 增加桌面 homography 點位到 9 到 16 點，並覆蓋整個會放硬幣的 ROI。
2. 點位盡量靠近實際會抓硬幣的區域，不要只標桌子一小塊。
3. 用尖的 TCP 指針或治具對點，不要用吸盤外緣估。
4. 每個點手臂停穩後再讀座標，避免拖曳或手動推動造成誤差。
5. Gemini 相機、桌子、手臂底座固定好，不要會晃。
6. 不要在標定後改 Gemini 解析度、原始畫面座標或相機安裝角度。
7. 桌面如果不平，或不同區域高度差明顯，要加入桌面平面模型，不能只用固定 Z。
8. 若要更精準下爪，加入 TCP offset / 吸盤中心偏移標定。
9. 每次正式使用前，用 3 個已知測試點做快速驗證，偏差超過 1 到 2 mm 就重標。
10. 目前只做 dry lower；加入真空後，還要標定吸盤中心、開關時序、吸取高度和放置位置。

## 硬幣分類規則

目前直徑分類：

- 1 元：20 mm
- 5 元：22 mm
- 10 元：26 mm
- 50 元：28 mm

系統會用量到的直徑接近哪個標準來分類。因為 1 元和 5 元差距小，邊緣不清楚、反光、傾斜、遮擋都會造成誤判。UI 會把可疑目標標成需要檢查。

## 常見問題

### 相機打不開，顯示硬體 MFT 缺少資源

通常是前一個程式還在佔用相機。

處理方式：

1. 關掉所有 UI、相機預覽、舊的 cmd。
2. 工作管理員檢查 Python 是否還在跑。
3. 重新開 `RUN_COIN_ROBOT_UI.bat`。

### 急停按了沒有反應

之前是 UI 忙碌時擋住按鈕，現在急停已改成可以獨立送出。若仍沒反應：

1. 先使用實體急停或 DobotStudio 停止。
2. 關掉目前 Python 程式。
3. 重啟 UI 後按 `清除報警並啟用手臂`。

### MG400 error 2 / 17 / 18 / 23 / 98

代表控制器拒絕或中斷動作，常見原因：

- 手臂還在 alarm / pause / disable 狀態。
- 目標超出安全範圍。
- 路徑被控制器拒絕。
- 選到靠工作區邊界的硬幣，尤其 `X` 太大或 `Y` 太靠邊。
- 低 Z dry lower 時碰撞偵測或姿態/路徑規劃觸發 alarm。
- 上一個動作沒有正常結束。

處理方式：

1. 看 UI 跳出的錯誤訊息。
2. 到 DobotStudio 看 alarm detail。
3. 清 alarm。
4. 確認手臂在安全位置。
5. 按 UI `清除報警並啟用手臂`。
6. 先測 hover，再測 dry lower。

## 目前還沒完成的部分

- 真空吸附 DO 尚未接入。
- 放置區位置與放置流程尚未正式完成。
- 手臂 Z 目前主要是固定桌面高度，不是完整 3D 表面追蹤。
- SAM3 模型目前仍以子程序方式載入，正式辨識可能需要數秒。若要更快，下一版應該把相機和模型常駐在 UI 背景 worker。
