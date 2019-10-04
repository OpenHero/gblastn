#! /bin/sh

# $Id: run_sybase_app.sh 123923 2008-04-08 14:36:53Z ivanov $
# Author:  Vladimir Ivanov, NCBI 
#
###########################################################################
# 
#  Run SYBASE application under MS Windows.
#  To run it under UNIX use configurable script "run_sybase_app.sh"
#  in build dir.
#
###########################################################################


if test -z "$SYBASE"; then
   SYBASE="C:\\Sybase"
   export SYBASE
fi
exec $CHECK_EXEC "$@"
