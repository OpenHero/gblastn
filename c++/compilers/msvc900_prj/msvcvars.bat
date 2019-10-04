@echo off
REM
REM $Id: msvcvars.bat 122871 2008-03-26 16:11:24Z gouriano $
REM

@if not "%VSINSTALLDIR%"=="" goto devenv
@call "%VS90COMNTOOLS%vsvars32.bat"

:devenv

if exist "%VS90COMNTOOLS%..\IDE\VCExpress.*" set DEVENV="%VS90COMNTOOLS%..\IDE\VCExpress"
if exist "%VS90COMNTOOLS%..\IDE\devenv.*" set DEVENV="%VS90COMNTOOLS%..\IDE\devenv"

:end
