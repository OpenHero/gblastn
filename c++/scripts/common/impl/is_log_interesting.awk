#!/usr/bin/awk -f
# $Id: is_log_interesting.awk 201243 2010-08-17 16:13:00Z ucko $

BEGIN { status = 1 }
/^[ 	]*$/                                                            { next }
/^(.*make\[[0-9]+\]: )?(Nothing to be done for `.*'|`.*' is updated)/   { next }
/^(dmake: defaulting to parallel mode|See the man page dmake.*)/        { next }
/make\[[0-9]+\]: (Enter|Leav)ing directory `.*'$/                       { next }
/^ln -f \..+\.dep .*\/status\/\..+\.dep$/                               { next }
# /^(.*make\[[0-9]+\]: \*\*\* No rule to make target .* needed by .*)/  { next }
# /^(.*make: Fatal error: Don't know how to make target `.*')/          { next }
{ status = 0; exit }
END { exit status }
