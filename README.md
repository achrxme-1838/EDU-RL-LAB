# w2_keep

2주차 실습 안전망 정책. 수강생 답이 틀려 15시 제어기 테스트를 못 할 때와 금요일 실기에 씀.
코드와 이력이 섞이지 않는 orphan 브랜치. 폴더마다 README.md에 run 이름, iteration, 학습 시간, 커밋, sha256.

- `tue/`, `wed/`, `thur/`: 그 요일 result 브랜치로 학습. 60·120·180분에 가장 가까운 checkpoint와 180분 `policy.onnx`
- `tue_nolin/`: 화요일 도전(`track_lin_vel_xy`) 없이 60분 (M3-b)
- `thur_nomotor/`: 목요일에서 motor DR만 끄고 60분 (M5-2)
- 재생: 그 요일 result 브랜치 작업 트리에서 `python scripts/rsl_rl/play.py --task Edu-A2-Velocity-v0 --checkpoint <model_N.pt 경로>`
  (화요일 정책은 입력 210이라 w2_result_tue에서만 재생됨)
