# 发布状态（2026-08-29 更新）

发布已完成，本文件转为运维备忘。

## 已完成

- [x] GitHub 仓库：https://github.com/SuperXionghaoren/FRetinalNet （main 分支，SSH 走 ssh.github.com:443）
- [x] Hugging Face 仓库：https://huggingface.co/waspwallbvb/FRetinalNet
  - dice_fdconv_drive_ori.pth（3.07 GB）、STAGE_FDCONV.pth（0.83 GB）均已上传并验证（HTTP 206）
  - 上传走了 hf-mirror.com（HF_ENDPOINT 环境变量），国内网络可用
- [x] Colab demo（demo.ipynb）：占位符已全部替换，本地端到端验证通过
- [x] 占位符替换：README/demo/scripts 中 HF=SuperXionghaoren→waspwallbvb、GH=SuperXionghaoren

## 待办 / 建议

- [ ] **撤销 HF 上传令牌**（已在聊天中暴露）：huggingface.co → Settings → Access Tokens → 删除 frtinalnet-upload
- [ ] 浏览器打开仓库页确认 README 渲染、点 Colab 徽章完整跑一遍
- [ ] （可选）GitHub 仓库 About 加 topics：retinal-vessel-segmentation / medical-imaging / pytorch / frequency-domain
- [ ] （可选）论文录用后在 README 加 Citation/bibtex
- [ ] （可选）后续更新权重：HF_TOKEN=... HF_ENDPOINT=https://hf-mirror.com huggingface-cli upload waspwallbvb/FRetinalNet <文件> <远端名>
