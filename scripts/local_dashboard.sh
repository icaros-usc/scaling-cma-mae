#!/bin/bash
# Dashboard for monitoring experiments in the local_logs/ directory.
#
# Usage:
#   scripts/local_dashboard.sh

print_header() {
  echo "------------- $1 -------------"
  echo
}

print_header "LOCAL LOGS"
i=0
for x in $(ls local_logs/); do
  d="local_logs/$x"
  i=$(($i + 1))
  experiment="$d/experiment.out"
  main_logdir=$(cat "$d/logdir")
  name=$(grep "experiment.name" "$main_logdir/config.gin" | sed "s/.*'\\(.*\\)'/\\1/g")
  seed=$(cat "$main_logdir/seed")

  echo "$i. $d ($name - Seed $seed)"
  echo "Logdir: $main_logdir"
  echo "Status: $(cat "$main_logdir/dashboard_status.txt")"
  echo "Tail: $(tail -n 1 "$experiment")"

  echo
done
