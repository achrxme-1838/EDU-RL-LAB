# tue

w2 실측 안전망 정책 (2026-11-19, Linux 학습 PC, RTX A5000). 학습 시간은 rsl_rl tot_time (Isaac Sim 시작 시간 제외).

| 시점 | 파일 | iteration | 학습 시간 | run 이름 | 커밋 (학습 시작 때) | sha256 |
|---|---|---|---|---|---|---|
| 60min | `model_1001.pt` | 1001 | 64.1분 | `2026-11-18_16-41-23_215908_w2_tue_full` | ddec55c (180분까지 이어 학습한 run은 462fc98, 학습 코드 같음) | `ada028900ad3c71728b07ebd72992c1a1b4aefa2bc56b0fa4649723c4d3a2a38` |
| 120min | `model_2001.pt` | 2001 | 125.4분 | `2026-11-18_16-41-23_215908_w2_tue_full` | ddec55c (180분까지 이어 학습한 run은 462fc98, 학습 코드 같음) | `7614257f7699fb4237ecdbd231ac4255df617d52f86569fff2847d7c38b872bc` |
| 180min | `model_2894.pt` | 2894 | 180.0분 | `2026-11-18_19-36-58_860207_w2_tue_full_to180` | ddec55c (180분까지 이어 학습한 run은 462fc98, 학습 코드 같음) | `f79bebdf0435293c51fc35edacd62b556dc7a0799830afa85498884cb5ec1f36` |

- `policy.onnx`: 180분 모델 `model_2894.pt`을 `play.py --checkpoint`로 내보낸 것. 입력 [1, 210], 출력 [1, 12]
- `policy.onnx` sha256: `d3cd1c8bfc89a32d97650c324c04a3e5f6774b37243152b0c65309df8d97da7e`
