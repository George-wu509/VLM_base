# 核心契約

Model forward 收 `input_ids [B,T]`、可選 `pixel_values [B,3,H,W]`、
`attention_mask [B,T]`、unshifted `labels [B,T]`。回傳完整 prefix/text
`logits [B,N+T,V]`、完整 labels、loss、image token 數 N。

SigLIP 224/16 產生 14×14=196 patch tokens；shuffle factor 2 將每個 2×2 block
合併成 channel，得到 49×3072；單一線性 projection 轉成 49×576。
SmolLM2 的 text embeddings 與 49 image embeddings 串接，由同一個 causal decoder 處理。
Tiny 32/8 的 16 patch tokens 同理得到 4 image prefix tokens。

Collator 將 `BOS + prompt + newline` 和 `response + EOS` 分開 tokenization，
準確遮罩 prompt。這不是 tokenizer chat template；推理也使用完全相同的 boundary。
model 在 image prefix 加 -100 labels，再將 logits[:-1] 對 labels[1:] 計算 CE。
第一個 answer token 由最後一個 prompt position 預測。EOS 可監督，padding 不可監督。
使用 EOS 當 padding 的 HF tokenizer 仍由 attention_mask/labels 區分兩者。

Frozen decoder 的 forward **不可放進 no_grad**，因為 projector 的梯度需要穿過 decoder。
Frozen vision encoder 可 no_grad；projector 對 vision features 仍可求梯度。
完全 frozen 的子模組置為 eval，避免 dropout 增加 alignment noise。

LoRA = frozen W + (B@A)×alpha/rank，初始 B=0，因此注入時不改變預測。
保存完整訓練模型而非單独 adapter，避免依賴不明確的 base checkpoint。
新階段 --init-from 先合併 LoRA，再按階段政策建立新 adapters。

DPO = -logsigmoid(beta × ((logπ(chosen)-logπ(rejected))
                          -(logref(chosen)-logref(rejected))))。
各 log-prob 是 answer/EOS tokens 的 sum，prompt、image、padding 被排除。
reference 複製該階段起始模型，eval/frozen，並在 resume 時從 checkpoint 還原，
不能把已更新的 policy 當成新 reference。DPO 要求所有 backbone/LoRA dropout 為零。

階段模組只有 policy：objective、train components、是否 LoRA。
Trainer 統一做 AMP、backward、accumulation、clip、step、schedule、checkpoint。
最後不足 accumulation 的一組按實際 microbatch 數平均。
目前每個 microbatch 的 mean loss 等權平均，不是整個 accumulation group 的
token-count weighted mean；如果 lengths 差异很大，應自行擴充相應 normalization。

資料抽樣由 seed+epoch 的獨立 generator 決定，DataLoader generator 與全域 dropout RNG
分開。Checkpoint 保存在 optimizer-step 邊界，不保存未完成的梯度。
Resume 重新生成該 epoch 的順序，跳過已完成 batch；所有 image transforms 都是 deterministic。
新加入隨機 augmentation 時，必須同步扩充 worker/augmentation RNG 還原契約。

Versioning 分成：固定上游 code SHA、HF model/tokenizer SHA、run 的 config/source/package/hash
快照、optimizer-step checkpoint、inference release bundle。Export 合併 LoRA並排除
optimizer/reference，tokenizer 仍隨模型一起保存。
