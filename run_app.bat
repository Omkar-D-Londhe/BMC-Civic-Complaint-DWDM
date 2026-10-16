@echo off
title BMC CivicLens Analytics Platform
echo =====================================================================
echo        BMC CIVICLENS ANALYTICS PLATFORM (FASTAPI WEB SERVER)
echo =====================================================================
echo Starting application on http://127.0.0.1:8000 ...
echo.
echo Open your browser at:
echo   - Executive Overview       : http://127.0.0.1:8000/overview
echo   - Complaint Explorer       : http://127.0.0.1:8000/explorer
echo   - OLAP Laboratory          : http://127.0.0.1:8000/olap
echo   - Supervised ML Models     : http://127.0.0.1:8000/ml-models
echo   - Patterns, Clusters & Rules: http://127.0.0.1:8000/patterns-and-clusters
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo =====================================================================
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload --reload-dir templates --reload-dir static --reload-dir src
pause
