#!/usr/bin/env bash
# Usage: .github/scripts/extract-notes.sh OWNER/SLUG GITREF

IFS='/'; read -ra REPOSITORY <<<"${1}"
OWNER=${REPOSITORY[0]}
SLUG=${REPOSITORY[1]}
GITREF=${2}

if [[ "${GITREF}" == refs/tags/* ]]; then
    VERSION="${GITREF#refs/tags/v}"
else
    VERSION="Unreleased"
fi

# Extract the release notes from the changelog
sed -n "/## \[${VERSION}\]/, /## /{ /##/!p }" CHANGELOG.md > notes.md

# Add a link to the full changelog
URL="https://github.com/${OWNER}/${SLUG}/blob/main/CHANGELOG.md"
echo "See [CHANGELOG.md](${URL}) for more details." >> notes.md

# Remove leading and trailing empty lines
sed -e :a -e '/./,$!d;/^\n*$/{$d;N;};/\n$/ba' -i notes.md
