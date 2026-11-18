# thur

w2 실측 안전망 정책 (2026-11-19, Linux 학습 PC, RTX A5000). 학습 시간은 rsl_rl tot_time (Isaac Sim 시작 시간 제외).

| 시점 | 파일 | iteration | 학습 시간 | run 이름 | 커밋 (학습 시작 때) | sha256 |
|---|---|---|---|---|---|---|
| 60min | `model_1501.pt` | 1501 | 67.1분 | `2026-11-18_16-40-21_483922_w2_thur_full` | 06478b7 | `57cdddb5e86f72cc9ec36ad5bd8e3c781347d4183bbf25ea61815cef32d7246c` |
| 120min | `model_2501.pt` | 2501 | 113.9분 | `2026-11-18_16-40-21_483922_w2_thur_full` | 06478b7 | `b72fc2a2faa30b900073806ca900db5ef20d0825cdfa1ae0c3d6344945ada04f` |
| 180min | `model_4001.pt` | 4001 | 184.5분 | `2026-11-18_16-40-21_483922_w2_thur_full` | 06478b7 | `270ba541abe933c42a0060a8fa8e46871459bd4eb5e570a4559494d89adf09e6` |

- `policy.onnx`: 180분 모델 `model_4001.pt`을 `play.py --checkpoint`로 내보낸 것. 입력 [1, 225], 출력 [1, 12]
- `policy.onnx` sha256: `7f46c690160d8212ef73e97d2c5979d5548ad33e7016cf3951c473adb3f8e86d`
