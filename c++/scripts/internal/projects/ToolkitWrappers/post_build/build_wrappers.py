#!/usr/bin/env python
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name of the Author nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id: build_wrappers.py 177690 2009-12-03 16:50:45Z satskyse $
#
# Author: Sergey Satskiy
#

""" Launch pad for the wrappers building scripts """

import os, sys, os.path
from optparse import OptionParser



def main():
    """ main entry point """

    parser = OptionParser( "%prog <version> <platform> <.../c++ directory> " \
                           "<lib directory> <install directory>" )
    parser.add_option( "-v", "--verbose", action="store_true", default=False,
                       help="be verbose", dest="verbose" )
    parser.add_option( "-c", "--skip-coverage", action="store_true",
                       default=False, help="skip coverage checks",
                       dest="skipCoverage" )
    options, args = parser.parse_args()

    if len( args ) != 5:
        return parserError( parser, "Incorrect number of arguments" )

    version, platform, cppDir, libDir, installDir = args
    if options.verbose:
        print "Package version:   " + version
        print "Platform:          " + platform
        print "C++ directory:     " + cppDir
        print "Lib directory:     " + libDir
        print "Install directory: " + installDir


    if 'Linux' in platform:
        return runBuildingOnUnix( version, \
                                  cppDir, libDir, installDir, \
                                  options.verbose, options.skipCoverage )
    if platform == "FreeBSD32":
        return notImplementedYet( platform )
    if platform == "IntelMAC":
        return notImplementedYet( platform )
    if platform == "SunOSSparc":
        return notImplementedYet( platform )
    if platform == "SunOSx86":
        return notImplementedYet( platform )
    if platform == "Win32":
        return notImplementedYet( platform )
    if platform == "Win64":
        return notImplementedYet( platform )
    if platform == "Win32_9":
        return notImplementedYet( platform )
    if platform == "Win64_9":
        return notImplementedYet( platform )

    print >> sys.stderr, "Unknown OS identifier: " + platform
    print >> sys.stderr, "Exiting post build script."
    return 1


def runBuildingOnUnix( version, cppDir, libDir, installDir,
                       verbose, skipCoverage ):
    """ Run wrappers building on UNIX """

    cmdLine = os.path.dirname( os.path.abspath( sys.argv[0] ) ) + \
              "/unix/build_unix_wrappers.py"
    if verbose:
        cmdLine += " -v"
    if skipCoverage:
        cmdLine += " -c"
    cmdLine += " " + version + " " + cppDir + " " + libDir + " " + installDir
    return os.system( cmdLine )


def parserError( parser, message ):
    """ Prints the message and help on stderr """

    sys.stdout = sys.stderr
    print message
    parser.print_help()
    return 1


def notImplementedYet( platform ):
    """ Prints a message """
    print >> sys.stderr, "Building on " + platform + \
                         " has not been implemented yet"
    return 1


# The script execution entry point
if __name__ == "__main__":
    retCode = main()
    if retCode != 0:
        retCode = 1
    sys.exit( retCode )

