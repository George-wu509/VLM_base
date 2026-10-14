# 本機驗證 — 2026-10-15（Asia/Taipei）

已在 `C:\LLM\VLM_base` 建立專案、本地 `.venv` 與 Git `main` repository。
程式尚未 commit，也未推送至遠端。執行環境：Python 3.12.14、PyTorch 2.6.0+cu124、
Transformers 4.51.3、NVIDIA RTX 4000 Ada Laptop GPU，11.99 GiB VRAM，BF16 可用。
完整實際套件版本記錄於 `requirements-windows-cu124.lock.txt`。

## 已執行

| 驗證 | 結果 | 本機產物 |
|---|---|---|
| 完整 unit / integration tests | **31 passed**；一個第三方 anyio deprecation warning | `artifacts/metrics/pytest-final.xml` |
| 原始 v0.1 reference integrity | 8 個原始檔案的 Git blob SHA1 全部符合 pinned tree | `vendor/nanovlm_v01/manifest.json` |
| Native vs Transformers 骨幹輸出，CPU FP32 | Vision / LM max absolute delta **均為 0.0**，單一固定測試輸入 | `artifacts/metrics/backbone-parity.json` |
| Tiny 全七階段，CPU FP32 | 每階段 3 optimizer steps，含評估/匯出 | `artifacts/runs/smoke-cpu/smoke_report.json` |
| Tiny 全七階段，CUDA BF16，最新程式 | 每階段 3 optimizer steps，含評估/匯出 | `artifacts/runs/smoke-verified/smoke_report.json` |
| 真實 SigLIP＋SmolLM2 alignment | 2 optimizer steps；只訓練 projector，loss 5.7187 → 5.4785 | `artifacts/runs/pretrained-alignment-check/` |
| 真實骨幹漸進 joint training | 1 optimizer step，decoder＋vision 最後 2 blocks＋projector | `artifacts/runs/pretrained-joint-check/` |
| 真實骨幹 spatial SFT | 加入 1152 spatial tokens，1 optimizer step，含 grounding annotation | `artifacts/runs/pretrained-spatial-sft-check/` |
| 真實 SFT checkpoint CLI 載入/推理 | 成功，tokenizer fingerprint 相符 | `scripts.infer` 實際執行 |
| API | TestClient 驗證 startup、health、image upload、generation、bad-image 400 | `tests/test_inference.py` |
| Python compile / dependencies | compileall 無錯誤；pip check 無相依衝突 | 本機執行 |

三次真實骨幹測量皆使用 batch 1、gradient accumulation 8、BF16、gradient checkpointing。
GPU allocator peak 是 **allocated memory**，不包含 CUDA driver/其他程式占用：

| 階段 | 可訓練參數 | Peak allocated |
|---|---:|---:|
| Alignment | Projector 1,769,472 | 1,229.4 MiB（約 1.20 GiB） |
| Joint | Vision 14,175,744 + decoder 134,515,008 + projector 1,769,472 | 3,065.3 MiB（約 2.99 GiB） |
| Spatial SFT | Decoder 135,178,560 + projector 1,769,472 | 3,019.0 MiB（約 2.95 GiB） |

以上使用少量 synthetic rectangles 與短文字，**不是長序列/正式資料的容量保證**。
新增 vocabulary 的 SFT loss 初始約 11.22，grounding 尚未學成；CLI 短生成也不代表
模型已具備 object detection 能力。Tiny heldout 的生成指標為零是預期的流程檢查結果。

## 重要測試契約

- image/prompt/padding 被排除；第一個 answer token 的 supervision 與 causal shift 正確。
- 改變未來 text tokens 不會影響較早 logits；right padding 不改變有效位置 logits。
- Projector 的梯度能穿過 frozen decoder；vision/decoder 在 alignment 完全不更新。
- LoRA merge 前後 logits 一致；DPO 起始 loss 接近 log(2)，reference 保持不變。
- Alignment、SFT（含 dropout）、DPO 中斷後續訓，與 uninterrupted CPU FP32 訓練
  的每個 state tensor **完全一致**。包括跨 epoch 與不足 accumulation 的最後一組。
- 影像內容 duplicate 與 group_id 不跨 train/val；影像 bytes 改變會被拒絕。
- checkpoint / export tokenizer 相符；壞的 tokenizer 替換會被拒絕。
- promotion gate 拒絕缺少指標、低於門檻或退步過多的候選。

## 未執行／不代表的能力

- 尚未使用正式 caption/VQA/grounding/preference dataset 訓練可用產品模型。
- LoRA/DPO 的完整流程已在 tiny CPU/CUDA 上驗證；真實骨幹未做完整 LoRA/DPO 品質實驗。
- Dockerfile 已提供，尚未建置/啟動 Docker container；API 沒有常駐背景啟動。
- 未執行大型公開 benchmark，也不提供 CIDEr、官方 VQAv2/POPE 或 segmentation 評分。
- Exact resume 測試限 CPU FP32；CUDA BF16 跑通流程，但不承諾跨 hardware/version bitwise 一致。

在 Codex 的 Windows sandbox 中，FastAPI TestClient 所需的本機 socketpair 會被阻擋，
因此完整 pytest 已於獲准的 sandbox 外執行；這不是模型或 API 程式錯誤。
一般 PowerShell 可直接執行 README 的 pytest 指令。
