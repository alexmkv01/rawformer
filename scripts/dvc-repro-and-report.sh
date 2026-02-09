#!/bin/sh
set -e -x

# Pull data and run cache from DVC remote
dvc pull --run-cache --allow-missing --force

{
    dvc repro
} || {
    printf "Saving partial output and artifacts\n"
    dvc push
    git add dvc.lock artifacts/
    git commit -m "[FAILED] dvc pipeline failed to complete [skip ci]"
    git push
    exit 1
}

echo "Saving reproduced output and artifacts"
dvc push

# Commit pipeline outputs
git add dvc.lock artifacts/
git commit --allow-empty -m "dvc pipeline reproduced [skip ci]"
git push

# Build report
git fetch --prune
{
    printf "## Metrics\n\n"
    dvc metrics diff main --md --all || printf "_Could not generate metrics diff_\n\n"
    printf "\n"
    printf "### Pipeline Status\n\n"
    dvc status || printf "All stages are up-to-date.\n"
} >comment.md

# Post report as PR comment
gh pr --repo "$GITHUB_REPOSITORY" \
    comment "$GITHUB_PR_NUMBER" \
    --body-file comment.md
