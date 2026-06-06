# Training Data Repository Design Prompt

Use this prompt in a new Codex/ChatGPT tab when designing a separate GitHub repository for pathology cell segmentation training data.

```text
Mình muốn thiết kế một GitHub repo riêng chỉ để store training data cho pathology cell segmentation.

Context:
- Hiện tại mình annotate trong QuPath.
- Sau khi annotate xong, mình export/update training data folder.
- Sau đó push training data lên GitHub.
- Khi có model/training pipeline mới, workflow sẽ là:
  1. Clone training data repo.
  2. Clone model/training repo.
  3. Point training code tới dataset đã clone.
  4. Train trên GPU cluster.

Goal:
Thiết kế cho mình một repo structure rõ ràng, dễ maintain, phù hợp cho nhiều model khác nhau như YOLO segmentation, CellSeg1, SAM/Cellpose future pipelines.

Yêu cầu:
- Tách rõ raw exports, curated datasets, model-specific dataset formats, metadata, manifests, and docs.
- Có versioning strategy cho dataset: ví dụ dataset releases, snapshot folders, hoặc manifest-based versioning.
- Có folder structure cụ thể.
- Có README template.
- Có update process sau mỗi vòng annotation: export từ QuPath -> validate -> update dataset -> commit/push.
- Có guidance cho GPU cluster: model repo clone data repo như thế nào.
- Có script suggestions: validate dataset, summarize counts, sync/export, make YOLO format, make CellSeg1 format.
- Recommend có nên dùng Git LFS không, và file nào nên/không nên đưa lên GitHub.
- Mục tiêu là repo training data độc lập, còn model code/training notebooks nằm ở repo khác.

Hãy đưa ra:
1. Proposed repo structure.
2. Dataset versioning strategy.
3. End-to-end workflow.
4. Example commands.
5. README skeleton.
6. Minimum scripts cần có.
7. Best practices để tránh push nhầm file lớn hoặc output training.
```
