#! /bin/sh

# $Id: cxx_filter.ICC.sh 117947 2008-01-23 16:59:05Z ucko $
# Author:  Dmitriy Beloslyudtsev, NCBI 
#
#  Filter out redundant warnings issued by Intel C++ (ICC) compiler.

tempfile="tmp$$"

"$@" 2>$tempfile

ret_code="$?"

cat $tempfile |
grep -v '^icc: NOTE: The evaluation period for this product ends on' |
grep -v '^conftest\.C:$' 1>&2

rm $tempfile

exit $ret_code
