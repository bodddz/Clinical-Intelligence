@echo off
title Push Clinical RAG Platform to GitHub
echo =========================================================================
echo  PUSHING TO GITHUB (https://github.com/bodddz/CTRL-ALT-CURE)
echo =========================================================================
echo.

git branch -M main
echo Pushing local commits to origin main...
git push -u origin main

if errorlevel 1 (
    echo.
    echo [NOTE] If non-fast-forward error occurs, pushing with overwrite:
    git push -u origin main --force
)

echo.
echo =========================================================================
echo  SUCCESS: Code pushed to GitHub. Vercel will auto-deploy.
echo =========================================================================
pause
