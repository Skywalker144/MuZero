#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s VERSION\n' "${0##*/}"
    printf 'Example: %s 1.0\n' "${0##*/}"
}

die() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

[[ $# -eq 1 ]] || {
    usage >&2
    exit 2
}

target_version=$1
[[ $target_version =~ ^[0-9]+([.][0-9]+)*$ ]] || die "invalid version '$target_version'; use a numeric version such as 0.10, 1.0, or 2"

command -v git >/dev/null 2>&1 || die 'git is required'
command -v perl >/dev/null 2>&1 || die 'perl is required'

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'not inside a Git repository'
[[ $(pwd -P) == $(cd "$repo_root" && pwd -P) ]] || die 'run this script from the Git repository root'

status=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n $status ]]; then
    printf 'Error: repository has uncommitted changes:\n%s\n' "$status" >&2
    exit 1
fi

prefix=${repo_root##*/}
[[ -f AGENTS.md ]] || die 'AGENTS.md does not exist in the repository root'

mapfile -t current_versions < <(
    grep -oE "${prefix}_V[0-9]+([.][0-9]+)*/" AGENTS.md \
        | sed -E "s/^${prefix}_V//; s:/$::" \
        | sort -u
)
[[ ${#current_versions[@]} -eq 1 ]] || die "AGENTS.md must identify exactly one current ${prefix}_V<version>/ directory"

source_version=${current_versions[0]}
source_dir="${prefix}_V${source_version}"
target_dir="${prefix}_V${target_version}"
[[ -d $source_dir ]] || die "current version directory '$source_dir' does not exist"
[[ ! -e $target_dir ]] || die "target '$target_dir' already exists"

IFS='.' read -r -a source_parts <<< "$source_version"
IFS='.' read -r -a target_parts <<< "$target_version"
component_count=${#source_parts[@]}
if (( ${#target_parts[@]} > component_count )); then
    component_count=${#target_parts[@]}
fi

target_is_greater=0
for ((index = 0; index < component_count; index++)); do
    source_component=${source_parts[index]:-0}
    target_component=${target_parts[index]:-0}
    while [[ ${#source_component} -gt 1 && ${source_component:0:1} == 0 ]]; do
        source_component=${source_component:1}
    done
    while [[ ${#target_component} -gt 1 && ${target_component:0:1} == 0 ]]; do
        target_component=${target_component:1}
    done
    if (( ${#target_component} > ${#source_component} )); then
        target_is_greater=1
        break
    fi
    if (( ${#target_component} < ${#source_component} )); then
        break
    fi
    if [[ $target_component > $source_component ]]; then
        target_is_greater=1
        break
    fi
    if [[ $target_component < $source_component ]]; then
        break
    fi
done
(( target_is_greater == 1 )) || die "target version $target_version must be greater than current version $source_version"

staging_root=$(mktemp -d "$repo_root/.new_version.XXXXXX")
staging_dir="$staging_root/$target_dir"
updated_agents="$staging_root/AGENTS.md"
target_created=0
completed=0

cleanup() {
    if (( target_created == 1 && completed == 0 )); then
        rm -rf -- "$repo_root/$target_dir"
    fi
    rm -rf -- "$staging_root"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$staging_dir"
copied_files=0
while IFS= read -r -d '' source_path; do
    relative_path=${source_path#"$source_dir/"}
    destination_path="$staging_dir/$relative_path"
    mkdir -p "${destination_path%/*}"
    cp -a -- "$repo_root/$source_path" "$destination_path"
    ((copied_files += 1))
done < <(git ls-files -z -- "$source_dir/")
(( copied_files > 0 )) || die "current version directory '$source_dir' has no tracked files"

updated_documents=0
while IFS= read -r -d '' document; do
    if grep -Fq "V$source_version" "$document"; then
        OLD_VERSION=$source_version NEW_VERSION=$target_version \
            perl -pi -e 's/\QV$ENV{OLD_VERSION}\E(?![0-9]|\.[0-9])/V$ENV{NEW_VERSION}/g' "$document"
        ((updated_documents += 1))
    fi
done < <(find "$staging_dir" -type f -iname '*.md' -print0)

cp -a -- AGENTS.md "$updated_agents"
OLD_VERSION=$source_version NEW_VERSION=$target_version \
    perl -pi -e 's/\QV$ENV{OLD_VERSION}\E(?![0-9]|\.[0-9])/V$ENV{NEW_VERSION}/g' "$updated_agents"
cmp -s AGENTS.md "$updated_agents" && die "AGENTS.md contains no current V$source_version reference to update"

trap '' HUP INT TERM
mv -- "$staging_dir" "$repo_root/$target_dir"
target_created=1
mv -- "$updated_agents" "$repo_root/AGENTS.md"
completed=1

git add AGENTS.md
git commit -m "Update AGENTS.md to V${target_version}"
git add .
git commit -m "Copy from V${source_version}"

printf 'Created %s from %s\n' "$target_dir" "$source_dir"
printf 'Copied tracked files: %d\n' "$copied_files"
printf 'Updated Markdown documents: %d\n' "$updated_documents"
printf 'Updated AGENTS.md to V%s\n' "$target_version"
