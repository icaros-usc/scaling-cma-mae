# Checks that the experiment configs are correct.
#
# Usage (from root dir of repo):
#   config/check.sh

set -e  # Exit if any of the commands fail.

for experiment in $(ls "config"); do
  case "$experiment" in
    algorithms|hpc|README.md|test.gin|check.sh|_legacy)
      echo "Skipping config/${experiment}"
      continue
      ;;
  esac

  for file in $(ls config/${experiment}); do
    fullname="config/${experiment}/${file}"

    case ${file} in
      ablations|me_es|shared.gin|sep_cma_mae_alpha_0-01.gin|sep_cma_mae_alpha_0-0.gin|sep_cma_mae_alpha_0-1.gin|sep_cma_mae_alpha_1-0.gin)
        echo "Skipping ${fullname}"
        continue
        ;;
      *)
        echo "Checking ${fullname}"
        readarray -t lines < "${fullname}" # -t excludes newlines.

        if [ "${lines[0]}" != "include \"config/${experiment}/shared.gin\"" ]; then
          echo "-> Bad shared.gin include"
          exit 1
        fi

        if [ "${lines[1]}" != "include \"config/algorithms/${file}\"" ]; then
          echo "-> Bad algorithm include"
          exit 1
        fi
    esac
  done
done
