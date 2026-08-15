@echo off
REM =============================================================================
REM DrugCLIP-WRN 一键运行脚本 (Windows)
REM =============================================================================
REM 用法:
REM   run.bat full      - 运行全流程
REM   run.bat train     - 仅训练模型
REM   run.bat screen    - 仅虚拟筛选
REM   run.bat dock      - 仅分子对接
REM   run.bat report    - 仅生成报告
REM =============================================================================

set MODE=%1
if "%MODE%"=="" set MODE=full

echo ============================================
echo DrugCLIP-WRN: WRN抑制剂虚拟筛选系统
echo 运行模式: %MODE%
echo ============================================

REM 激活conda环境
call conda activate drugclip 2>nul || echo 警告: 使用当前Python环境

python "%~dp0main.py" --mode %MODE%

echo ============================================
echo 运行完成
echo ============================================
pause
