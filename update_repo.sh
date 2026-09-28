#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

# Official Enzyme repo
UPSTREAM="upstream"
UPSTREAM_BRANCH="main"

# Your fork
ORIGIN="origin"

MAIN_BRANCH="main"

# Add/remove fix branches here.
# Each branch contributes ALL commits absent from upstream, including reverts.
# The atan2 fix is already upstream (#3222); the obsolete atan2 revert must
# also be removed from mixu/support-clang-21-cuda, not just from this list.
FIX_BRANCHES=(
    "mixu/support-clang-21-cuda"
)
WORKING_BRANCH="mixu/linux/combined-fix"

# # add "windows-msvc-cuda-build-fixes" to FIX_BRANCHES and change the name of WORKING_BRANCH
# FIX_BRANCHES+=("windows-msvc-cuda-build-fixes")
# WORKING_BRANCH="mixu/windows/combined-fix"

# ============================================================
# Helpers
# ============================================================

die() {
    echo
    echo "ERROR: $1"
    echo
    exit 1
}

branch_exists() {
    git show-ref --verify --quiet "refs/heads/$1"
}


# ============================================================
# Initial checks
# ============================================================

echo
echo "============================================================"
echo " Updating Enzyme + local fixes"
echo "============================================================"
echo

git rev-parse --git-dir >/dev/null 2>&1 \
    || die "Not inside a git repository."

# Make sure working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree is not clean. Commit or stash your changes first."
fi

# Check upstream
if ! git remote get-url "$UPSTREAM" >/dev/null 2>&1; then
    die "Remote '$UPSTREAM' does not exist.

Run:

    git remote add upstream https://github.com/EnzymeAD/Enzyme.git"
fi

# Check origin
if ! git remote get-url "$ORIGIN" >/dev/null 2>&1; then
    die "Remote '$ORIGIN' does not exist."
fi

echo "Official Enzyme:"
echo "    $(git remote get-url "$UPSTREAM")"

echo
echo "Your fork:"
echo "    $(git remote get-url "$ORIGIN")"


# ============================================================
# 1. Fetch latest official Enzyme
# ============================================================

echo
echo "============================================================"
echo " Fetching latest upstream"
echo "============================================================"

git fetch "$UPSTREAM"

echo
echo "Latest upstream commit:"
git log -1 --oneline "$UPSTREAM/$UPSTREAM_BRANCH"


# ============================================================
# 2. Update local main
# ============================================================

echo
echo "============================================================"
echo " Updating $MAIN_BRANCH"
echo "============================================================"

git switch "$MAIN_BRANCH"

git merge --ff-only "$UPSTREAM/$UPSTREAM_BRANCH"

echo
echo "$MAIN_BRANCH is now:"
git log -1 --oneline


# ============================================================
# 3. Rebase every fix branch onto latest upstream
# ============================================================

for FIX_BRANCH in "${FIX_BRANCHES[@]}"; do

    echo
    echo "============================================================"
    echo " Rebasing: $FIX_BRANCH"
    echo "============================================================"

    if ! branch_exists "$FIX_BRANCH"; then
        die "Local branch '$FIX_BRANCH' does not exist."
    fi

    git switch "$FIX_BRANCH"

    if ! git rebase "$UPSTREAM/$UPSTREAM_BRANCH"; then

        echo
        echo "============================================================"
        echo " REBASE CONFLICT"
        echo "============================================================"
        echo
        echo "Branch:"
        echo "    $FIX_BRANCH"
        echo
        echo "Resolve conflicts, then:"
        echo
        echo "    git add <files>"
        echo "    git rebase --continue"
        echo
        echo "Or abort:"
        echo
        echo "    git rebase --abort"
        echo
        echo "Then run this script again."
        echo

        exit 1
    fi

done


# ============================================================
# 4. Re-create combined-fix from latest upstream
# ============================================================

echo
echo "============================================================"
echo " Rebuilding $WORKING_BRANCH"
echo "============================================================"

git switch "$MAIN_BRANCH"

# Create branch if it doesn't exist,
# or reset it if it already exists.
git switch -C "$WORKING_BRANCH" "$UPSTREAM/$UPSTREAM_BRANCH"


# ============================================================
# 5. Apply every fix branch
# ============================================================

for FIX_BRANCH in "${FIX_BRANCHES[@]}"; do

    echo
    echo "------------------------------------------------------------"
    echo " Applying: $FIX_BRANCH"
    echo "------------------------------------------------------------"

    mapfile -t COMMITS < <(
        git rev-list \
            --reverse \
            "$UPSTREAM/$UPSTREAM_BRANCH..$FIX_BRANCH"
    )

    if [ "${#COMMITS[@]}" -eq 0 ]; then
        echo "No additional commits found in $FIX_BRANCH"
        continue
    fi

    echo "Commits to apply:"

    for COMMIT in "${COMMITS[@]}"; do
        git log -1 --oneline "$COMMIT"
    done

    echo

    if ! git cherry-pick "${COMMITS[@]}"; then

        echo
        echo "============================================================"
        echo " CHERRY-PICK CONFLICT"
        echo "============================================================"
        echo
        echo "While applying:"
        echo "    $FIX_BRANCH"
        echo
        echo "Resolve conflicts, then:"
        echo
        echo "    git add <files>"
        echo "    git cherry-pick --continue"
        echo
        echo "Or abort:"
        echo
        echo "    git cherry-pick --abort"
        echo

        exit 1
    fi

done


# ============================================================
# 6. Show final combined branch
# ============================================================

echo
echo "============================================================"
echo " Combined branch created successfully"
echo "============================================================"

echo
echo "Branch:"
echo "    $WORKING_BRANCH"

echo
echo "Contains:"
echo
echo "    $UPSTREAM/$UPSTREAM_BRANCH"

for FIX_BRANCH in "${FIX_BRANCHES[@]}"; do
    echo "    + $FIX_BRANCH"
done

echo
echo "Recent history:"
echo

git log \
    --oneline \
    --decorate \
    --graph \
    -20


# ============================================================
# 7. Push combined-fix to YOUR fork
# ============================================================

echo
echo "============================================================"
echo " Pushing $WORKING_BRANCH to your fork"
echo "============================================================"

echo
echo "Destination:"
echo "    $(git remote get-url "$ORIGIN")"
echo

git push \
    --force-with-lease \
    "$ORIGIN" \
    "$WORKING_BRANCH:$WORKING_BRANCH"


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo " SUCCESS"
echo "============================================================"
echo
echo "Local:"
echo "    $WORKING_BRANCH"
echo
echo "Remote:"
echo "    $ORIGIN/$WORKING_BRANCH"
echo
echo "Your combined branch is now pushed to:"
echo "    $(git remote get-url "$ORIGIN")"
echo
