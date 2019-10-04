@echo off
REM
REM $Id: msvcvars.bat 196747 2010-07-08 14:00:02Z gouriano $
REM

@if not "%VSINSTALLDIR%"=="" goto devenv
@call "%VS100COMNTOOLS%vsvars32.bat"

:devenv

if exist "%VS100COMNTOOLS%..\IDE\VCExpress.*" set DEVENV="%VS100COMNTOOLS%..\IDE\VCExpress"
if exist "%VS100COMNTOOLS%..\IDE\devenv.*" set DEVENV="%VS100COMNTOOLS%..\IDE\devenv"

:end
