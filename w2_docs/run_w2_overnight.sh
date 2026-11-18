#!/usr/bin/env bash
# w2 실측 연속 실행 (실측_지시서.md "학습 시간 계획"). Linux 학습 PC에서 Isaac Lab 환경을 켠 상태로 실행.
#
#   export EDU=~/isaac_ws/EDU-RL-LAB      # 1-2절의 작업 트리(w2_edu, w2_result_*)가 $EDU/.. 아래에 있어야 함
#   export KEEP=~/w2_keep                 # 로그와 결과를 모을 곳
#   nohup bash run_w2_overnight.sh > ~/w2_run.out 2>&1 &              # 전부 (약 11–12시간)
#   nohup bash run_w2_overnight.sh thur thur_nomotor m8 > ... &         # 일부 단계만 (PC 여러 대로 나눌 때)
#
# 단계 (생략하면 이 순서로 전부): m1 check tue tue_nolin wed thur thur_nomotor m8
# 바꿀 수 있는 값: MIN_FULL(기본 180분), MIN_SHORT(60분), LOGGER(wandb|tensorboard)
# 한 단계가 실패해도 다음 단계로 넘어감. 진행 기록은 $KEEP/logs/run.log, 단계별 출력은 $KEEP/logs/<이름>.txt

set -u
: "${EDU:?EDU를 EDU-RL-LAB clone 경로로 export 할 것}"
KEEP=${KEEP:-$HOME/w2_keep}
W=$(cd "$EDU/.." && pwd)
T=Edu-A2-Velocity-v0
MIN_FULL=${MIN_FULL:-180}
MIN_SHORT=${MIN_SHORT:-60}
LOGGER=${LOGGER:-wandb}
mkdir -p "$KEEP/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$KEEP/logs/run.log"; }

run() {  # run <이름> <폴더> <명령...>
  local name=$1 dir=$2; shift 2
  log "시작 $name ($dir): $*"
  (cd "$dir" && "$@") > "$KEEP/logs/$name.txt" 2>&1
  local rc=$?
  log "끝 $name (종료 코드 $rc)"
  return $rc
}

iters_for() {  # iters_for <폴더> <이름> <분>: 100 iteration으로 속도를 재고, <분>에 들어가는 iteration 수를 출력
  local dir=$1 name=$2 min=$3
  run "${name}_speed" "$dir" python scripts/rsl_rl/train.py --task $T --headless --max_iterations 100 \
    --run_name "${name}_speed" --logger tensorboard >&2
  python - "$KEEP/logs/${name}_speed.txt" "$min" <<'EOF'
import re, sys
t = [float(x) for x in re.findall(r"Iteration time:\s*([\d.]+)s", open(sys.argv[1], errors="ignore").read())][-20:]
print(int(float(sys.argv[2]) * 60 / (sum(t) / len(t))) if t else 0)
EOF
}

train() {  # train <이름> <폴더> <분>
  local name=$1 dir=$2 min=$3 n
  n=$(iters_for "$dir" "$name" "$min")
  if [ -z "$n" ] || [ "$n" -le 0 ]; then log "$name: 속도 측정 실패, 건너뜀"; return 1; fi
  log "$name: ${min}분 = $n iteration"
  echo "$name ${min}min $n $(date '+%F %T')" >> "$KEEP/logs/iterations.txt"
  run "$name" "$dir" timeout $((min + 20))m python scripts/rsl_rl/train.py --task $T --headless \
    --max_iterations "$n" --run_name "$name" --logger "$LOGGER"
}

variant() {  # variant <폴더 이름> <브랜치> <지울 블록 정규식>: 커밋하지 않는 임시 작업 트리를 만들어 cfg 블록 하나를 지움
  local dir="$W/$1" branch=$2 pattern=$3
  if [ ! -d "$dir" ]; then git -C "$EDU" worktree add --detach "$dir" "$branch" >&2 || return 1; fi
  git -C "$dir" checkout -q -- . || return 1
  python - "$dir/edu_rl_lab/tasks/locomotion/a2_env_cfg.py" "$pattern" <<'EOF' || return 1
import re, sys
path, pattern = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
out, k = re.subn(pattern, "", src, count=1, flags=re.M)
if k != 1:
    sys.exit("지울 블록을 찾지 못함: " + pattern)
open(path, "w", encoding="utf-8").write(out)
EOF
  echo "$dir"
}

step_m1() {
  for d in tue wed thur; do run "m1_check_$d" "$W/w2_edu" python scripts/w2_check.py --day $d; done
  run m1_train "$W/w2_edu" python scripts/rsl_rl/train.py --task $T --headless --max_iterations 30 \
    --run_name w2_edu_m1 --logger tensorboard
}

step_check() {
  for d in tue wed thur; do run "check_$d" "$W/w2_result_$d" python scripts/w2_check.py --day $d; done
  run smoke_thur "$W/w2_result_thur" python tests/smoke_a2.py
}

step_tue()  { train w2_tue_full  "$W/w2_result_tue"  "$MIN_FULL"; }
step_wed()  { train w2_wed_full  "$W/w2_result_wed"  "$MIN_FULL"; }
step_thur() { train w2_thur_full "$W/w2_result_thur" "$MIN_FULL"; }

step_tue_nolin() {  # 화요일 도전(track_lin_vel_xy)을 쓰지 않은 상태 (M3-b)
  local d
  d=$(variant w2_tmp_tue_nolin w2_result_tue '^    track_lin_vel_xy = RewTerm\(\n.*\n    \)\n') || return 1
  run check_tue_nolin "$d" python scripts/w2_check.py --day tue
  train w2_tue_nolin "$d" "$MIN_SHORT"
}

step_thur_nomotor() {  # motor DR을 끈 상태 (M5-2)
  local d
  d=$(variant w2_tmp_thur_nomotor w2_result_thur '^    actuator_gains = EventTerm\(\n(?:.*\n)*?    \)\n') || return 1
  train w2_thur_nomotor "$d" "$MIN_SHORT"
}

step_m8() {
  for n in 4096 2048 1024; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$KEEP/logs/m8_env${n}_gpu_mib.txt" &
    local smi=$!
    run "m8_env$n" "$W/w2_result_thur" python scripts/rsl_rl/train.py --task $T --headless --num_envs $n \
      --max_iterations 50 --run_name "w2_thur_env$n" --logger tensorboard
    kill $smi 2>/dev/null
    log "m8 env$n GPU 메모리 최대 $(sort -n "$KEEP/logs/m8_env${n}_gpu_mib.txt" | tail -1) MiB"
  done
}

STEPS=${*:-m1 check tue tue_nolin wed thur thur_nomotor m8}
log "단계: $STEPS (MIN_FULL=$MIN_FULL, MIN_SHORT=$MIN_SHORT, LOGGER=$LOGGER, 작업 트리 위치 $W)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$KEEP/logs/run.log"
for s in $STEPS; do
  "step_$s" || log "단계 $s 실패 (다음 단계로 진행)"
done
log "모두 끝. iteration 기록: $KEEP/logs/iterations.txt"
