#!/bin/bash

# Script to clone EmbodiedMinds repository and checkout the 'sameer' branch

REPO_URL_HTTPS="https://github.com/Oxlelouch/EmbodiedMinds.git"
REPO_URL_SSH="git@github.com:Oxlelouch/EmbodiedMinds.git"
BRANCH="sameer"

echo "Attempting to clone repository..."

# Try HTTPS first
if git clone -b $BRANCH $REPO_URL_HTTPS . 2>/dev/null; then
    echo "✓ Successfully cloned repository using HTTPS"
elif git clone -b $BRANCH $REPO_URL_SSH . 2>/dev/null; then
    echo "✓ Successfully cloned repository using SSH"
else
    echo "✗ Failed to clone. Trying alternative method..."
    
    # Alternative: Clone main branch first, then checkout
    if git clone $REPO_URL_HTTPS . 2>/dev/null || git clone $REPO_URL_SSH . 2>/dev/null; then
        echo "✓ Cloned main branch, now checking out 'sameer' branch..."
        git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH origin/$BRANCH 2>/dev/null
        echo "✓ Checked out 'sameer' branch"
    else
        echo "✗ Error: Could not clone repository."
        echo ""
        echo "Possible issues:"
        echo "1. Repository is private - you need to authenticate"
        echo "2. Repository URL might be incorrect"
        echo "3. You don't have access to this repository"
        echo ""
        echo "To authenticate with GitHub:"
        echo "  - For HTTPS: Use GitHub CLI (gh auth login) or use a personal access token"
        echo "  - For SSH: Make sure your SSH key is added to GitHub"
        exit 1
    fi
fi

echo ""
echo "Repository contents:"
ls -la

echo ""
echo "Current branch:"
git branch

echo ""
echo "✓ Setup complete! You can now work on the incomplete code files."

