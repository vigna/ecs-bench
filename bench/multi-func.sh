#!/bin/bash
#
# Parallel (multi-bench) runs of VFunc and PHastR + array on an AWS
# m8a.metal-24xl, with two builds: "vec" (default flags) and "novec" (loop
# vectorizer off, SLP vectorizer on). The machine writes the results to
# ~/results-multi-func/{vec,novec}, with the same layout as the other
# results (<method>/multibench.*.out).
#
#   bench/multi-func.sh launch <host>        copy the sources to the machine and
#                                            start the runs, detached
#   bench/multi-func.sh watch <host> <dir>   copy the results to <dir> every ten
#                                            minutes, until the runs end or the
#                                            machine cannot be reached
#   bench/multi-func.sh run                  what the machine runs
#
# The runs use the same parameters as the existing parallel results (96
# query threads, 96 construction threads, 10^8 queries per thread, 2^33
# keys in total) and the same compiler (the default of a new machine is
# now a later version). A size with no result (e.g., a construction that
# ran out of memory) is run again with fewer construction threads, which
# changes only the construction time. Results that exist are not run
# again, so the script can be started again after a failure.
#
# Test on another machine, at a small scale and without the checks:
#
#   CHECK=0 T=4 TOTAL=23 Q=2000000 OUT=/tmp/out TARGETS=/tmp bench/multi-func.sh run

set -euo pipefail

TOOLCHAIN=${TOOLCHAIN:-1.99.1}
INSTANCE=${INSTANCE:-m8a.metal-24xl}
T=${T:-96}                  # query threads and construction threads
TOTAL=${TOTAL:-33}          # log2 of the total number of keys
MIN=${MIN:-20}              # log2 of the smallest size
Q=${Q:-100000000}           # queries per thread
RETRY_B=${RETRY_B:-16}      # construction threads of the second attempt
CHECK=${CHECK:-1}           # 0 skips the machine checks and the setup
OUT=${OUT:-$HOME/results-multi-func}
TARGETS=${TARGETS:-$HOME}   # where the two cargo target directories go
METHODS=VFunc,PHastR
NOVEC_FLAGS="-C target-cpu=native -C llvm-args=-vectorize-loops=false"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=ecs-bench-multi

log() { echo "$(date -u +%FT%TZ) $*"; }

# Prints the instance type of an EC2 machine.
instance_type() {
	local token
	token=$(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token \
		-H "X-aws-ec2-metadata-token-ttl-seconds: 60")
	curl -s -m 3 -H "X-aws-ec2-metadata-token: $token" \
		http://169.254.169.254/latest/meta-data/instance-type
}

# Stops if the machine is not the right one, or if it is in use.
check_machine() {
	local type load
	type=$(instance_type)
	[ "$type" = "$INSTANCE" ] || { log "Instance type is '$type', not $INSTANCE"; return 1; }
	[ "$(nproc)" = "$T" ] || { log "$(nproc) CPUs, not $T"; return 1; }
	load=$(cut -d' ' -f1 /proc/loadavg)
	awk -v l="$load" 'BEGIN { exit !(l < 2) }' || { log "The machine is in use (load $load)"; return 1; }
}

# Installs what the runs need (not GCC: BuRR is not run) and sets the CPU
# frequency as aws.sh does.
set_up_machine() {
	command -v cc >/dev/null && command -v cpupower >/dev/null || sudo dnf -y install gcc kernel-tools python3
	if ! command -v rustup >/dev/null && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
			sh -s -- -y --default-toolchain "$TOOLCHAIN" --profile minimal
	fi
	# shellcheck disable=SC1091
	. "$HOME/.cargo/env"
	rustup toolchain install "$TOOLCHAIN" --profile minimal
	export RUSTUP_TOOLCHAIN=$TOOLCHAIN
	rustc --version | grep -q "^rustc $TOOLCHAIN " || { log "Wrong compiler: $(rustc --version)"; return 1; }

	sudo cpupower frequency-set --governor performance
	sudo cpupower frequency-set --min 4.50GHz
	sudo cpupower frequency-set --max 4.50GHz
	sudo cpupower idle-set -D 0

	# The construction of a PHastR from a lender keeps 2^14 partition
	# buffers for each structure, and from 2^27 keys each buffer has its
	# own memory mapping. With several structures under construction the
	# process goes over the default limit on memory mappings (65530), and
	# allocations fail with most of the memory free.
	sudo sysctl -w vm.max_map_count=16777216
}

# Records what was run.
record_environment() {
	{
		echo "date: $(date -u +%FT%TZ)"
		echo "sources: $(cat "$ROOT/VERSION" 2>/dev/null || echo "$(git -C "$ROOT" rev-parse HEAD) (git)")"
		echo "rustc: $(rustc --version)"
		echo "kernel: $(uname -r)"
		echo "cpus: $(nproc)"
		echo "parameters: T=$T TOTAL=$TOTAL MIN=$MIN Q=$Q RETRY_B=$RETRY_B"
		if [ "$CHECK" = 1 ]; then
			echo "instance: $(instance_type)"
			echo "memory: $(free -g | awk '/Mem/ { print $2 " GB" }')"
			echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
			echo "frequency: $(cpupower -c 0 frequency-info -p | sed -n 2p | xargs)"
			echo "max_map_count: $(sysctl -n vm.max_map_count)"
		fi
		echo
	} >>"$OUT/ENVIRONMENT"
}

# Runs the harness on a variant ($1) with the given construction threads
# ($2) and more arguments.
harness() {
	local v=$1 b=$2
	shift 2
	python3 "$ROOT/bench/run.py" --results-dir "$OUT/$v" multi-bench --query-threads "$T" \
		--build-threads "$b" --total-log2 "$TOTAL" --queries "$Q" "$@"
}

# Builds a variant ($1), checks its compiler flags, and runs all sizes.
sweep() {
	local v=$1 expected flags l m n present
	export CARGO_TARGET_DIR=$TARGETS/target-multi-$v
	if [ "$v" = novec ]; then
		export RUSTFLAGS=$NOVEC_FLAGS
		expected="['-C', 'target-cpu=native', '-C', 'llvm-args=-vectorize-loops=false']"
	else
		unset RUSTFLAGS
		expected="['-C', 'target-cpu=native']"
	fi

	log "Building variant $v"
	(cd "$ROOT/multi-vfilter" && cargo build --profile bench)
	flags=$(python3 -c "
import glob, json
f = sorted(glob.glob('$CARGO_TARGET_DIR/release/.fingerprint/multi-vfilter-*/bin-multi-vfilter.json'))
print(json.load(open(f[-1]))['rustflags'])")
	[ "$flags" = "$expected" ] || { log "Variant $v has flags $flags"; return 1; }
	if command -v objdump >/dev/null; then
		log "Variant $v: $(objdump -d "$CARGO_TARGET_DIR/release/multi-vfilter" | grep -c vpgather || true) gather instructions in the binary"
	fi

	log "Running variant $v"
	harness "$v" "$T" --method "$METHODS" --min-log2 "$MIN"

	# For tests of the second attempt only
	if [ -n "${TEST_DROP:-}" ]; then
		# shellcheck disable=SC2086
		rm -f "$OUT/$v"/$TEST_DROP
	fi

	# The sizes without a result are run again with fewer construction threads
	present=$(harness "$v" "$T" --method "$METHODS" --min-log2 "$MIN" --report)
	for ((l = MIN; l <= TOTAL; l++)); do
		n=$((1 << l))
		for m in ${METHODS//,/ }; do
			if ! echo "$present" | awk -F'\t' -v m="$m" -v n="$n" '$1 == m && $4 == n { f = 1 } END { exit !f }'; then
				log "Variant $v: no result for $m with 2^$l keys; second attempt with $RETRY_B construction threads"
				harness "$v" "$RETRY_B" --method "$m" --min-log2 "$l" --max-log2 "$l" || true
			fi
		done
	done
}

run() {
	mkdir -p "$OUT"
	echo RUNNING >"$OUT/STATUS"
	trap '[ "$(cat "$OUT/STATUS")" = DONE ] || echo FAILED >"$OUT/STATUS"' EXIT
	if [ "$CHECK" = 1 ]; then
		check_machine
		set_up_machine
	fi
	record_environment
	sweep vec
	sweep novec
	log "Results of the first attempts:"
	for v in vec novec; do
		echo "== $v"
		harness "$v" "$T" --method "$METHODS" --min-log2 "$MIN" --report
	done
	echo DONE >"$OUT/STATUS"
	log "Done."
}

launch() {
	local host=$1 version
	# The machine must be the right one, and nobody must be using it
	ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" "INSTANCE=$INSTANCE T=$T bash -s check" <"$0"
	version=$(mktemp)
	{
		echo "ecs-bench $(git -C "$ROOT" rev-parse HEAD) [$(git -C "$ROOT" status --short -- bench multi-vfilter | tr '\n' ' ')]"
		echo "sux-rs $(git -C "$ROOT/sux-rs" rev-parse HEAD) [$(git -C "$ROOT/sux-rs" status --short | tr '\n' ' ')]"
	} | tr '\n' ' ' >"$version"
	ssh -o BatchMode=yes "$host" "mkdir -p $REMOTE"
	rsync -a --delete --exclude target --exclude .git --exclude __pycache__ \
		"$ROOT/bench" "$ROOT/multi-vfilter" "$ROOT/sux-rs" "$host:$REMOTE/"
	rsync -a "$version" "$host:$REMOTE/VERSION"
	rm -f "$version"
	ssh -o BatchMode=yes "$host" \
		"nohup setsid bash $REMOTE/bench/multi-func.sh run >multi-func.log 2>&1 </dev/null & echo started"
}

watch() {
	local host=$1 dest=$2 failures=0 status
	mkdir -p "$dest"
	while :; do
		if rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$host:results-multi-func/" "$dest/" 2>/dev/null; then
			failures=0
			status=$(cat "$dest/STATUS" 2>/dev/null || true)
			if [ "$status" = DONE ] || [ "$status" = FAILED ]; then
				rsync -a -e "ssh -o BatchMode=yes" "$host:multi-func.log" "$dest/" || true
				echo "$status"
				tail -n 40 "$dest/multi-func.log" || true
				[ "$status" = DONE ]
				return
			fi
		else
			failures=$((failures + 1))
			if [ "$failures" -ge 3 ]; then
				echo "UNREACHABLE: three checks in a row failed"
				return 2
			fi
		fi
		sleep 600
	done
}

case "${1:-}" in
	run) run ;;
	check) check_machine ;;
	launch) launch "$2" ;;
	watch) watch "$2" "$3" ;;
	*)
		sed -n '3,25p' "$0"
		exit 1
		;;
esac
