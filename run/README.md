# Run Folder Guide

這個資料夾已整理成三區：

- `operator/`：一般操作，只放主程式入口。
- `calibration/`：重新框 ROI、兩相機對位、手臂桌面座標標定。
- `engineering/`：工程測試、舊流程、乾跑測試、Gemini 調整工具。

日常只需要：

```text
operator/開啟主程式.bat
```

常用標定：

```text
calibration/重新框選ROI.bat
calibration/拍攝雙相機同步棋盤.bat
calibration/計算畫質到深度相機對位.bat
calibration/手臂桌面座標標定.bat
```

標定拍照或手臂教點啟動時會先詢問：

```text
1 = 清空舊資料重新開始（會先自動備份，不直接刪除）
2 = 保留舊資料，繼續新增
```

如果是重新標定，通常選 `1`，避免新舊資料混在一起造成誤差很大。

如果只是正常撿硬幣流程，請優先用主 UI，不要直接點 `engineering/` 裡的手臂乾跑測試。
