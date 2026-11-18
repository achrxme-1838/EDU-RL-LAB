# wed

w2 실측 안전망 정책 (2026-11-19, Linux 학습 PC, RTX A5000). 학습 시간은 rsl_rl tot_time (Isaac Sim 시작 시간 제외).

| 시점 | 파일 | iteration | 학습 시간 | run 이름 | 커밋 (학습 시작 때) | sha256 |
|---|---|---|---|---|---|---|
| 60min | `model_1001.pt` | 1001 | 53.2분 | `2026-11-18_16-40-41_111248_w2_wed_full` | 153af23 (180분까지 이어 학습한 run은 766dd3a, 학습 코드 같음) | `e27886f63af078bd7aabc223450ea2eef40f5bc6af570590d6397df3a2540f3c` |
| 120min | `model_2501.pt` | 2501 | 121.8분 | `2026-11-18_16-40-41_111248_w2_wed_full` | 153af23 (180분까지 이어 학습한 run은 766dd3a, 학습 코드 같음) | `095561dd165184364d65ca99d5d91d34fe6afdf67d5226b10b15f26c64224feb` |
| 180min | `model_3773.pt` | 3773 | 180.4분 | `2026-11-18_19-25-58_789293_w2_wed_full_to180` | 153af23 (180분까지 이어 학습한 run은 766dd3a, 학습 코드 같음) | `607a60d53eaf25ab5bed378e8a6854c830a6b481fa117bf88f47a7a0bb0a2fa1` |

- `policy.onnx`: 180분 모델 `model_3773.pt`을 `play.py --checkpoint`로 내보낸 것. 입력 [1, 225], 출력 [1, 12]
- `policy.onnx` sha256: `0791bc28056ad03bc68f7c13ba7aa04d96a9541d8f43fa5c8d77f1335e2570c4`
