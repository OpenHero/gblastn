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
# $Id: build_unix_wrappers.py 257036 2011-03-09 20:30:45Z satskyse $
#
# Author: Sergey Satskiy
#

""" Build C++ Toolkit wrappers on UNIX """

import os, sys, os.path, re, tempfile, datetime
from optparse import OptionParser
from subprocess import Popen, PIPE
from traceback import format_tb
import py_compile


# disable pylint message about exception handling
# pylint: disable-msg=W0702
# pylint: disable-msg=W0704

def main():
    """ main entry point """

    parser = OptionParser( "%prog <version> <.../c++ directory> " \
                           "<library directory> <install directory>" )
    parser.add_option( "-v", "--verbose", action="store_true", default=False,
                       help="be verbose", dest="verbose" )
    parser.add_option( "-c", "--skip-coverage", action="store_true",
                       default=False, help="skip coverage checks",
                       dest="skipCoverage" )
    options, args = parser.parse_args()

    if len( args ) != 4:
        return parserError( parser, "Incorrect number of arguments" )

    version, cppDir, libDir, installDir = args
    cppDir = os.path.abspath( cppDir )
    if not cppDir.endswith( "/" ):
        cppDir += "/"

    libDir = os.path.abspath( libDir )
    if not libDir.endswith( "/" ):
        libDir += "/"

    buildDir = libDir.replace( "/lib/", "/build/" )

    installDir = os.path.abspath( installDir )
    if not installDir.endswith( "/" ):
        installDir += "/"

    if options.verbose:
        print "Building C++ Toolkit wrappers on UNIX"
        print "Package version:   " + version
        print "Source directory:  " + cppDir
        print "Library directory: " + libDir
        print "Build directory:   " + buildDir
        print "Install directory: " + installDir

    if not os.path.exists( cppDir ):
        print >> sys.stderr, "Cannot find c++ directory. " \
                             "Expected it here: " + cppDir
        return 2

    if not os.path.exists( libDir ):
        print >> sys.stderr, "Cannot find library directory. " \
                             "Expected it here: " + libDir
        return 2

    if not os.path.exists( buildDir ):
        print >> sys.stderr, "Cannot find build directory. " \
                             "Expected it here: " + buildDir
        return 2

    if not os.path.exists( installDir ):
        print >> sys.stderr, "Cannot find install directory. " \
                             "Expected it here: " + installDir
        return 2


    # Build a list of all the toolkit header files
    headers = []
    toolkitHeaders( cppDir + "include", headers )
    headers.sort()


    # cd to the directory where the wrappers are going to be built
    wrappersTopDir = cppDir + "src/internal/cppcore/toolkit_wrappers/"
    if not os.path.exists( wrappersTopDir ):
        print >> sys.stderr, "Cannot find wrappers project directory. " \
                             "Expected it here: " + wrappersTopDir
        return 2
    currentDir = os.getcwd()
    os.chdir( wrappersTopDir )

    retCode = 0
    try:

        # Run c pre processor on all the toolkit header files
        preprocessAllHeaders( headers, buildDir, cppDir )

        # Postprocess the single file after pre processor
        postprocessAllHeaders( cppDir )

        # Generate version.ig file
        generateVersionFile( version )

        # Run SWIG first time to generate the ncbi.py file
        print "Erasing ncbi_templates.ig..."
        os.system( 'echo "" > ncbi_templates.ig' )
        print "Erasing cobject_declarations.ig..."
        os.system( 'echo "" > cobject_declarations.ig' )

        print "Running SWIG first time to generate ncbi.py..."
        runSWIG( cppDir + "include/", buildDir, options.verbose, True )

        # Run templates extractor
        print "Extracting templates..."
        if os.system( './extract_templates.py ' \
                      'postprocessed ncbi.py > ncbi_templates.ig' ) != 0:
            raise Exception( "Error running templates generator" )
        # Create COBJECT_DECLARATION(...)..
        print "Extracting CObjects..."
        if os.system( './extract_cobjects.py > cobject_declarations.ig' ) != 0:
            raise Exception( 'Error running cobjects extractor' )

        print "AI: insert %template where required..."
        cmdLine = './insert_templates.py postprocessed ' \
                  'templates_to_insert swig.log'
        if options.verbose:
            cmdLine += ' -v'
        cmdLine += ' > postprocessed.ins'
        if os.system( cmdLine ) != 0:
            raise Exception( 'Error inserting template directives' )

        os.system( 'mv postprocessed postprocessed.orig' )
        os.system( 'mv postprocessed.ins postprocessed' )

        # AI: insert the inclusion of the collected NCBI templates to the
        # suitable place in the postprocessed file
        insertNcbiTemplates()

        # Run SWIG again. This time it will pick up the generated templates
        # instatinations
        print "Running SWIG second time..."
        runSWIG( cppDir + "include/", buildDir, options.verbose, False )

        # injecting run-time python version check
        print "Generating run-time python version check..."
        generateVersionCheckScript()

        # Make adjustments in the generated C++ code
        print "Making adjustments in the generated C++ code..."
        adjustCpp()

        # Replace the 0-length .cpp file in the wrapper project dir and run make
        buildWrapperLibrary( buildDir, options.verbose )

        # build ncbipy
        buildNCBIpy( buildDir, options.verbose )

        # Copy the final libraries to the location where they are picked up
        # by the release framework to be sent back to the client
        copyReadyLibraries( libDir, installDir )

        # Execute tests
        executeTests( installDir )

        # Run splitter
        cmdLine = "./split_py.py ncbi.py"
        if options.verbose:
            cmdLine += ' -v'
        if os.system( cmdLine ) != 0:
            raise Exception( "Error splitting ncbi.py" )

        # Compile ncbi_auto.py and ncbi_fast.py
        py_compile.compile( 'ncbi_auto.py', doraise = True )
        py_compile.compile( 'ncbi_fast.py', doraise = True )
        py_compile.compile( 'check_version.py', doraise = True )

        # copy the splitted splitted classes
        print "Copying splitted classes..."
        retCode = os.system( "cp -rp ncbi_classes/ " + installDir + "python_wrappers" )
        retCode += os.system( "cp -p ncbi_auto.py* " + installDir + "python_wrappers" )
        retCode += os.system( "cp -p ncbi_fast.py* " + installDir + "python_wrappers" )
        retCode += os.system( "cp -p check_version.py* " + installDir + "python_wrappers" )
        if retCode != 0:
            raise Exception( "Error copying splitted classes" )

        if not options.skipCoverage:
            cmdLine = "./check_coverage.py " + installDir
            if options.verbose:
                cmdLine += ' -v'
            if os.system( cmdLine ) != 0:
                raise Exception( "Error checking coverage" )

    except:
        retCode = 1
        print >> sys.stderr, getExceptionInfo()

    os.chdir( currentDir )
    return retCode


def generateVersionCheckScript():
    " Generates the run-time version check into ncbi.py "

    pVersion = 'v.' + str( sys.version_info[0] ) + '.' + str( sys.version_info[1] )
    errorMsg1 = '"Warning: Your python interpreter version ' \
                '(v." + str(version_info[0]) + "." + str(version_info[1]) + ") ' \
                'does not match the python version (' + pVersion + ') ' \
                'that was used to generate the wrappers."'
    errorMsg2 = '"This may lead to a performance slowdown and/or to run-time errors."'
    errorMsg3 = '"Use the wrappers at your own risk."'

    # Generate the check_version.py script
    f = open( 'check_version.py', 'w' )
    f.write( '# Automatically generated\n' )
    f.write( 'from sys import version_info\n' )
    f.write( 'import sys\n' )
    f.write( 'if version_info[0] != ' + str( sys.version_info[0] ) + ' or ' + \
             'version_info[1] != ' + str( sys.version_info[1] ) + ':\n' )
    f.write( '    print >> sys.stderr, ' + errorMsg1 + '\n' )
    f.write( '    print >> sys.stderr, ' + errorMsg2 + '\n' )
    f.write( '    print >> sys.stderr, ' + errorMsg3 + '\n' )
    f.close()

    return


def executeTests( installDir ):
    """ Run tests for the generated wrappers """

    print "Running tests..."

    retCode = 0
    for lang in getTargetLanguages():
        targetDir = installDir + lang + "_wrappers/tests"
        sourceDir = os.path.abspath( lang + "_wrappers/tests" )

        if not os.path.exists( sourceDir ):
            print "No tests found for " + lang + ". Skipping..."
            continue

        if os.path.exists( targetDir ):
            if os.system( "rm -rf " + targetDir ) != 0:
                print "Cannot remove directory '" + targetDir + "'. Skipping..."
                retCode += 1
                continue
        os.mkdir( targetDir )

        # copy the tests
        if os.system( "cp -rp " + sourceDir + "/* " + targetDir ) != 0:
            print "Error copying tests to the package directory. Skipping..."
            retCode += 1
            continue

        # Execution is language specific
        if lang == "python":
            if executePythonTests( installDir, targetDir ) != 0:
                retCode += 1
            continue
        print "Test execution for " + lang + " has not been implemented yet."
        retCode += 1

    if retCode != 0:
        raise Exception( "Error running tests" )
    return


def executePythonTests( installDir, targetDir ):
    """ Experimental """

    retCode = 0

    # Search for all the python files in the test directory
    testCode = []
    for item in os.listdir( targetDir ):
        if item.endswith( ".py" ):
            testCode.append( targetDir + "/" + item )

    runDir = installDir + "python_wrappers"
    envVar = "PYTHONPATH=" + runDir
    ncbipy = runDir + "/ncbipy"
    for item in testCode:
        cmdLine = "cd " + runDir + " && " + envVar + " " + ncbipy + " " + item
        cmdLine += " > " + item.replace( ".py", ".log" ) + " 2>&1"
        print "Running " + os.path.basename( item ) + " ..."
        if os.system( cmdLine ) != 0:
            print "Error test executing. Command line: " + cmdLine
            retCode += 1

    return retCode


def insertNcbiTemplates():
    """ Insert the '%include ncbi_templates.ig' to the appropriate place
        in the postprocessed header file. The idea is to find the end of
        the ncbiobj.hpp file ends """

    print "Inserting the collected templates to the postprocessed file..."

    # Find out what the last # ...ncbiobj.hpp line number is
    output = safeRun( [ 'grep', '-n', "include/corelib/ncbiobj.hpp",
                        'postprocessed' ] ).split( '\n' )
    lastIndex = len(output) - 1
    while True:
        if len( output[lastIndex] ) == 0:
            lastIndex -= 1
        else:
            break

    lastNcbiObjLine = int(output[ lastIndex ].split(':')[0])

    # Open the postprocessed file and skip all the lines till lastNcbiObjLine
    fin = open( "postprocessed" )
    fout = open( "postprocessed.ins", "w" )

    lineNumber = 1
    tail = False
    for line in fin:
        if lineNumber <= lastNcbiObjLine:
            # skip all the lines before the certain one
            fout.write( line )
            lineNumber += 1
            continue

        if tail:
            # The rest of the file lines
            fout.write( line )
            continue

        if line.startswith( '#' ):
            # Insertion point
            fout.write( '%include ncbi_templates.ig\n' )
            tail = True
        fout.write( line )

    fout.close()
    fin.close()

    # swap files
    os.system( "rm -rf postprocessed" )
    os.system( "mv postprocessed.ins postprocessed" )
    return



def postprocessAllHeaders( cppDir ):
    """ postrocess the file after C pre processor """

    print "Postprocessing the single header file..."


    # Read the excludes from the single header file
    excludes = []
    if os.path.exists( "postprocess_excludes" ):
        f = open( "postprocess_excludes", "r" )
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith( "#" ):
                continue
            excludes.append( re.compile( line ) )
        f.close()


    fin = open( "includeall" )
    fout = open( "postprocessed", "w" )

    filesStack = []
    isFirst = True
    isSkip = False
    for line in fin:
        if isFirst:
            isFirst = False
            lineNum, fName, flags = parsePreprocessorLine( line )
            filesStack.append( [fName, False] )

        # Special case: they come from system headers
        if line.startswith( '#define FILENAME_MAX' ) or \
           line.startswith( '#define PATH_MAX' ) or \
           line.startswith( '#define __CHAR_BIT__' ) or \
           line.startswith( '#define CHAR_BIT' ):
            fout.write( line )
            continue

        # Suppress some preprocessor output caused by the -dD option key
        if isSkip:
            if not line.startswith( "# 1" ):
                continue
            isSkip = False
        if line.startswith( "#" ):
            if "<built-in>" in line or "<command line>" in line or "<command-line>" in line:
                isSkip = True
                continue
        if line.startswith( "#define __flexarr []" ):
            continue

        if not line.startswith( "#" ) or line.startswith( "#pragma" ) \
           or line.startswith( "#define" ) or line.startswith( "#undef" ):
            if not filesStack[ len(filesStack)-1 ][1]:
                fout.write( line )      # write it if it should not be excluded
            continue

        # This is a preprocessor line
        lineNum, fName, flags = parsePreprocessorLine( line )
        if 1 in flags:
            # print "Pushing: " + fName + " Size: " + str(len(filesStack))
            filesStack.append( [fName, shouldExclude(excludes, fName, flags)] )
            fout.write( line )
            continue

        if 2 in flags:
            # print "Popping: " + filesStack[ len(filesStack)-1 ][0] + \
            #       " Size: " + str(len(filesStack)-1)
            filesStack = filesStack[ :-1 ]
            fout.write( line )
            continue

        fout.write( line )

    fout.close()
    fin.close()

    return


def shouldExclude( excludes, fileName, flags ):
    """ Experimental """

    # 1 is beginning of a file
    if not 1 in flags:
        return False


    leaveThemIn = [ 'connect/services/grid_worker_app_impl.hpp',
                    'corelib/impl/ncbi_dbsvcmapper.hpp' ]


    # Special case - objmgr and serial implementation
    if 'objmgr/impl/' in fileName or \
       'serial/impl/' in fileName:
        return False

    # Specifically known to be left
    for item in leaveThemIn:
        if item in fileName:
            return False

    # Specifically known files to be excluded
    for excl in excludes:
        if excl.match( fileName ):
            return True

    # Standard headers usually come from /usr/...
    if fileName.startswith( '/usr/' ):
        return True

    if '/c++/' in fileName:
        if '/include/' in fileName or '/inc/' in fileName:
            # This is a toolkit header
            return False

    return True


def parsePreprocessorLine( line ):
    """ Experimental """

    parts = line.split( '"' )
    fileName = parts[1]

    noFileLine = line.replace( '"' + fileName + '"', '' )
    parts = noFileLine.split()

    lineNum = int( parts[1] )
    flags = []
    index = 2
    while index < len( parts ):
        flags.append( int( parts[index] ) )
        index += 1
    return lineNum, fileName, flags


def preprocessAllHeaders( headers, buildDir, cppDir ):
    """ runs c pre processor on all the toolkit header files """

    print "Preprocessing all the toolkit header files..."
    # Pick up the -D keys which were used to build the toolkit libraries
    makefileInPath = buildDir + "Makefile.mk"
    fakeMake = buildDir + "fakemake"

    # A map between targets and variables
    makefileContent = { 'compiler' : 'CONF_CXX',
                        'cppflags' : 'CONF_CPPFLAGS',
                        'boost_include' : 'BOOST_INCLUDE',
                        'compress_include' : 'CMPRS_INCLUDE',
                        'hdf5_include' : 'HDF5_INCLUDE',
                        'sqlite3' : 'SQLITE3_INCLUDE' }

    f = open( fakeMake, "w" )
    f.write( "include " + buildDir + "Makefile.mk\n" )
    for key in makefileContent.keys():
        f.write( key + ":\n" )
        f.write( "\techo $(" + makefileContent[ key ] + ")\n" )
    f.close()

    compiler = safeRun( [ "make", "-s", "-f", fakeMake, "compiler" ] ).strip()

    compilerKeys = ""
    for key in makefileContent.keys():
        if key == 'compiler':
            continue
        line = safeRun( [ "make", "-s", "-f", fakeMake, key ] ).strip()
        if len(line) > 0:
            compilerKeys += " " + line

    os.system( "rm -rf " + fakeMake )

    # Exclude some headers
    patterns = []
    try:
        f = open( "ignore_toolkit_headers", "r" )
        for line in f:
            line = line.strip()
            if line == "" or line.startswith( "#" ):
                continue
            patterns.append( re.compile( line ) )
            print "Exclude filter found: " + line
        f.close()
    except:
        raise Exception( "Error processing 'ignore_toolkit_headers' file." )

    # Debug support
    includePatterns = []
    if os.path.exists( 'include_only' ):
        try:
            includeOnlyFile = open( "include_only", "r" )
            for line in includeOnlyFile:
                line = line.strip()
                if line == "" or line.startswith( "#" ):
                    continue
                includePatterns.append( re.compile( line ) )
                print "Debug support: include only filter: " + line
            includeOnlyFile.close()
        except:
            raise Exception( "Error processing 'include_only' file." )


    # Form a file with all the #include directives
    allHeadersFile = os.path.abspath( "includeall.hpp" )
    f = open( allHeadersFile, "w" )
    for item in headers:
        header = item.replace( cppDir + "include/", "" )
        matched = False
        for pattern in patterns:
            if pattern.match( header ):
                matched = True
                break
        if not matched:
            if len( includePatterns ) > 0:
                # Debugging - check first if matched required headers
                matchedInc = False
                for incPattern in includePatterns:
                    if incPattern.match( header ):
                        matchedInc = True
                        break
                if matchedInc:
                    f.write( '#include "' + header + '"\n' )
                else:
                    print "Toolkit header to be excluded: " + header
            else:
                f.write( '#include "' + header + '"\n' )
        else:
            print "Toolkit header to be excluded: " + header

    f.close()


    # make a link to the generated file
    linkName = cppDir + 'include/includeall.hpp'
    if not os.path.exists( linkName ):
        os.system( "ln -s " + allHeadersFile + " " + linkName )


    #cmdLine = compiler + " -E -x c++-header " + compilerKeys
    cmdLine = compiler + " -E -dD -x c++-header " + compilerKeys

    # Make the CTempString to make a copy
    cmdLine += " -DNCBI_TEMPSTR_USE_A_COPY"
    # Exclude some toolkit code fragments
    cmdLine += " -DNCBI_SWIG"
    cmdLine += " " + allHeadersFile + " > includeall"
    print "Running preprocessor on all the found toolkit headers: " + cmdLine

    if os.system( cmdLine ) != 0:
        raise Exception( "Error preprocessing toolkit headers" )
    return

    # Now let's extract defines from the toolkit headers
    #cmdLine = compiler + " -E -dD -x c++-header " + compilerKeys
    #cmdLine += " -DNCBI_TEMPSTR_USE_A_COPY"
    #cmdLine += " -DNCBI_SWIG"
    #cmdLine += " " + allHeadersFile + " > raw_defines"
    #print "Running preprocessor"


def buildNCBIpy( buildDir, verbose ):
    """ builds ncbipy if python is selected """

    if not "python" in getTargetLanguages():
        return

    print "Building ncbipy..."

    # Replace the source to the real one
    srcFileName = "python_wrappers/ncbipy.c.real"
    tgtFileName = "python_wrappers/ncbipy.c"

    if not os.path.exists( tgtFileName + ".bak" ):
        os.system( "cp " + tgtFileName + " " + tgtFileName + ".bak" )

    os.system( "cp -f " + srcFileName + " " + tgtFileName )

    # Run building the ncbipy
    cmdLine = 'cd ' + buildDir + '; make CXX_WRAPPER="" -f Makefile.flat ncbipy.exe'
    if verbose:
        print "ncbipy building command line: " + cmdLine

    if os.system( cmdLine ) != 0:
        raise Exception( "Error making ncbipy." )

    return


def copyReadyLibraries( libDir, installDir ):
    """
    copies the ready libraries to the location where they are picked up
    and put into the final tar.gz
    """

    print "Copying the built libraries..."

    for lang in getTargetLanguages():
        srcFile = libDir + "lib" + lang + "_wrappers-dll.so"
        if not os.path.exists( srcFile ):
            raise Exception( "Wrapper library is not found for " + lang + \
                             ". Expected here: " + srcFile )

        targetDir = installDir + lang + "_wrappers/"
        if not os.path.exists( targetDir ):
            os.mkdir( targetDir )
        if not os.path.isdir( targetDir ):
            raise Exception( "Cannot create " + lang + " wrappers directory." )
        if lang == "python":
            dstFile = targetDir + "_ncbi.so"
            os.system( "strip " + srcFile )
            os.system( "cp " + srcFile + " " + dstFile )
            os.system( "cp ncbi.py " + targetDir )
            os.system( "cp check_version.py " + targetDir )

            ncbipyFile = libDir.replace( "/lib/", "/bin/" ) + "ncbipy"
            os.system( "strip " + ncbipyFile )
            os.system( "cp " + ncbipyFile + " " + targetDir )

            # Compile the python code
            os.system( "touch " + targetDir + "__init__.py" )
            os.system( "cd " + targetDir + "; ./ncbipy -c 'import ncbi'" )
            os.system( "cd " + targetDir + "; ./ncbipy -c 'import __init__'" )

    return


def generateVersionFile( version ):
    """ generates a file with a version constant """

    try:
        f = open( 'version.ig', 'w' )
        f.write( '/*\n' )
        f.write( ' * Auto generated file\n' )
        f.write( ' */\n' )
        f.write( '\n' )
        f.write( '%constant version = "built ' + \
                 datetime.datetime.now().ctime() + ", v." + version + '";\n' )
        f.close()
    except:
        raise Exception( "Error generating version.ig file." )
    return


def adjustCpp():
    """ Makes adjustments in the generated c++ file """


    enums = ['EREG_Storage', 'EMT_Lock' ]
    replacements = [
        (' ETagClass ', ' CAsnBinaryDefs::ETagClass '),
        (' ETagConstructed ', ' CAsnBinaryDefs::ETagConstructed '),
        (' TLongTag ', ' CAsnBinaryDefs::TLongTag '),
        (' TTypeCreate ', ' ncbi::CTypeInfo::TTypeCreate '),
        ('ncbi::blast::std::vector', 'std::vector'),
        ('ncbi::blast::std::string', 'std::string'),
        ('ncbi::objects::CRPCClient', 'ncbi::CRPCClient'),
        ('ncbi::blast::std::list', 'std::list'),
        (' SVectorElement ',
            ' ncbi::cobalt::CSparseKmerCounts::SVectorElement '),
        (' STreeLeaf ', ' ncbi::cobalt::CTree::STreeEdge '),
        (' STreeEdge ', ' ncbi::cobalt::CTree::STreeEdge '),
        (' SAlignStats ', ' ncbi::CContigAssembly::SAlignStats '),
        (' SSegment ', ' ncbi::CNWFormatter::SSegment '),
        (' SAlignedCompartment *', ' ncbi::CSplign::SAlignedCompartment *'),
        (' Fasta2CdParams *', ' ncbi::cd_utils::CCdFromFasta::Fasta2CdParams *'),
        ('(Fasta2CdParams const',
            '(ncbi::cd_utils::CCdFromFasta::Fasta2CdParams const'),
        (' Range *', ' struct_util::Block::Range *'),
        ('(Range *', '(struct_util::Block::Range *'),
        (' SProgress *', ' ncbi::cobalt::CMultiAligner::SProgress *'),
        (' CCdCore::USE_PENDING_ALIGNMENT', ' ncbi::cd_utils::CCdCore::USE_PENDING_ALIGNMENT'),
        (' SSeqIdChooser *', ' CSeq_align::SSeqIdChooser *'),
        ('(SSeqIdChooser *)', '(CSeq_align::SSeqIdChooser *)'),
        (' SBlobAccessDescr *', ' ICache::SBlobAccessDescr *'),
        ('(SBlobAccessDescr *)', '(ICache::SBlobAccessDescr *)'),
        (' SBlobData *', ' CNetCacheClient::SBlobData *'),
        ('(SBlobData *)', '(CNetCacheClient::SBlobData *)'),
        (' SAlignment_Row *', ' SAlignment_Segment::SAlignment_Row *'),
        ('SAlignment_Row const &', 'SAlignment_Segment::SAlignment_Row const &'),
        ('(SAlignment_Row *)', '(SAlignment_Segment::SAlignment_Row *)'),
        ('< CSeqData >', '< ncbi::blastdbindex::CSequenceIStream::CSeqData >'),
        (' SOptions result;', ' ncbi::blastdbindex::CDbIndex::SOptions result;'),
        ('(new SOptions(', '(new ncbi::blastdbindex::CDbIndex::SOptions('),
        ('const SOptions&', 'const ncbi::blastdbindex::CDbIndex::SOptions&'),
        (' SOptions *', ' ncbi::blastdbindex::CDbIndex::SOptions *'),
        ('(SOptions const &)', '(ncbi::blastdbindex::CDbIndex::SOptions const &)'),
        (' SSearchData *', ' ncbi::dbindex_search::CSRSearch::SSearchData *'),
        ('(SSearchData const &)', '(ncbi::dbindex_search::CSRSearch::SSearchData const &)'),
        (' SSearchOptions *', ' ncbi::blastdbindex::CDbIndex::SSearchOptions *'),
        ('(SSearchOptions const &)', '(ncbi::blastdbindex::CDbIndex::SSearchOptions const &)'),
        (' C_Id *', ' ncbi::objects::CId_pat_Base::C_Id *'),
        ('(C_Id const &)', '(ncbi::objects::CId_pat_Base::C_Id const &)'),
        (' SBlastDbParam *', ' ncbi::objects::CBlastDbDataLoader::SBlastDbParam *'),
        ('(SBlastDbParam const &)', '(ncbi::objects::CBlastDbDataLoader::SBlastDbParam const &)'),
        (' SParam *', ' ncbi::objects::CDataLoaderPatcher::SParam *'),
        ('(SParam const &)', '(ncbi::objects::CDataLoaderPatcher::SParam const &)'),
        (' SConnInfo result;', ' ncbi::objects::CReaderServiceConnector::SConnInfo result;'),
        ('(new SConnInfo(', '(new ncbi::objects::CReaderServiceConnector::SConnInfo('),
        ('const SConnInfo&', 'const ncbi::objects::CReaderServiceConnector::SConnInfo&'),
        (' SConnInfo *', ' ncbi::objects::CReaderServiceConnector::SConnInfo *'),
        ('< CBlobStream >', '< ncbi::objects::CWriter::CBlobStream >'),
        ('(new SObjectDescr(', '(new ncbi::objects::CLDS_Query::SObjectDescr('),
        ('const SObjectDescr&', 'const ncbi::objects::CLDS_Query::SObjectDescr&'),
        (' SObjectDescr result;', ' ncbi::objects::CLDS_Query::SObjectDescr result;'),
        (' SObjectDetails *', ' ncbi::objects::CLDS_CoreObjectsReader::SObjectDetails *'),
        ('(SObjectDetails *)', '(ncbi::objects::CLDS_CoreObjectsReader::SObjectDetails *)'),
        ('SAlignment_Row const &', 'SAlignment_Segment::SAlignment_Row const &'),
        (' FeatureInfo *', ' ncbi::objects::CDisplaySeqalign::FeatureInfo *'),
        (' AlnInfo *', ' ncbi::objects::CVecscreen::AlnInfo *'),
        ('ncbi::objects::CDisplaySeqalign_FeatureInfo', 'ncbi::objects::CDisplaySeqalign::FeatureInfo'),
        (' TSegTypeFlags',' ncbi::objects::CAlnMap::TSegTypeFlags'),
        (' TSignedRange *', ' ncbi::objects::CAlnMap::TSignedRange *'),
        ('(TSignedRange *)', '(ncbi::objects::CAlnMap::TSignedRange *)'),
        (' TNumrow', ' ncbi::objects::CAlnMap::TNumrow'),
        (' TNumchunk', ' ncbi::objects::CAlnMap::TNumchunk'),
        (' CAlnChunk ', ' ncbi::objects::CAlnMap::CAlnChunk '),
        ('(CAlnChunk *)','(ncbi::objects::CAlnMap::CAlnChunk *)'),
        (' TConn ', ' ncbi::objects::CReaderRequestResult::TConn '),
        (' EStatus', ' ncbi::CThreadPool_Task::EStatus'),
        (' SMismatchResultsEntry *', ' ncbi::dbindex_search::CSRSearch::SMismatchResultsEntry *'),
        ('(SMismatchResultsEntry *', '(ncbi::dbindex_search::CSRSearch::SMismatchResultsEntry *'),
        (' TSRResults', ' ncbi::dbindex_search::CSRSearch::TSRResults'),
        ('(TSRResults *', '(ncbi::dbindex_search::CSRSearch::TSRResults *'),
        (' ELevel', ' ncbi::dbindex_search::CSRSearch::ELevel'),
        ('< SResultData >', '< ncbi::dbindex_search::CSRSearch::SResultData >'),
        (' TSeqNum', ' ncbi::dbindex_search::CSRSearch::TSeqNum'),
        (' CPattern *', ' ncbi::cobalt::CMultiAlignerOptions::CPattern *'),
        ('(CPattern const &)', '(ncbi::cobalt::CMultiAlignerOptions::CPattern const &)'),
        (' TMasterMethod', ' ncbi::cd_utils::CCdFromFasta::TMasterMethod'),
        (' stat ', ' struct stat '),
        (' stat(', ' struct stat('),
        (' stat&', ' struct stat&'),
        ('(TMapValue *)', '(ncbi::CIntervalTreeTraits::TMapValue *)'),
        (' TChunkId', ' ncbi::objects::CTSE_Info_Object::TChunkId'),
        (' TChunkIds *', ' ncbi::objects::CTSE_Info_Object::TChunkIds *'),
        ('(TChunkIds const &)', '(ncbi::objects::CTSE_Info_Object::TChunkIds const &)'),
        ('ncbi::std::', 'std::'),
        ('ncbi::blast::std::set', 'std::set'),
        (' SSbsArrays *', ' ncbi::blast::CGumbelParamsResult::SSbsArrays *'),
        ('(SSbsArrays *)', '(ncbi::blast::CGumbelParamsResult::SSbsArrays *)'),
        (' CRowIterator *', ' ncbi::CQuery::CRowIterator *'),
        ('(CRowIterator const &)', '(ncbi::CQuery::CRowIterator const &)'),
        ('(CRowIterator *)', '(ncbi::CQuery::CRowIterator *)'),
        (' CRowIterator result;', ' ncbi::CQuery::CRowIterator result;'),
        ('new CRowIterator', 'new ncbi::CQuery::CRowIterator'),
        ('const CRowIterator&', 'const ncbi::CQuery::CRowIterator&'),
        (' CField *', ' ncbi::CQuery::CField *'),
        ('(CField *)', '(ncbi::CQuery::CField *)'),
        (' TOffsetPair *', ' ncbi::CSeqDB::TOffsetPair *'),
        ('(TOffsetPair *)', '(ncbi::CSeqDB::TOffsetPair *)'),
        ('(CIdSet *)', '(ncbi::CWinMaskUtil::CIdSet *)'),
        (' TFeatIdIntList arg2 ;', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdIntList arg2 ;'),
        (' TFeatIdIntList *', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdIntList *'),
        (' TFeatIdIntList result;', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdIntList result;'),
        ('new TFeatIdIntList', 'new ncbi::objects::CTSE_Chunk_Info::TFeatIdIntList'),
        ('const TFeatIdIntList&', 'const ncbi::objects::CTSE_Chunk_Info::TFeatIdIntList&'),
        ('< TResult >', '< ncbi::blast::CGumbelParamsResult::TResult >'),

        (' TFeatIdStrList arg2 ;', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdStrList arg2 ;'),
        (' TFeatIdStrList *', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdStrList *'),
        (' TFeatIdStrList result;', ' ncbi::objects::CTSE_Chunk_Info::TFeatIdStrList result;'),
        ('new TFeatIdStrList', 'new ncbi::objects::CTSE_Chunk_Info::TFeatIdStrList'),
        ('const TFeatIdStrList&', 'const ncbi::objects::CTSE_Chunk_Info::TFeatIdStrList&'),

        (' TFeatArray arg2 ;', ' ncbi::objects::feature::CFeatTree::TFeatArray arg2 ;'),
        (' TFeatArray *', ' ncbi::objects::feature::CFeatTree::TFeatArray *'),
        (' TFeatArray result;', ' ncbi::objects::feature::CFeatTree::TFeatArray result;'),
        ('new TFeatArray', 'new ncbi::objects::feature::CFeatTree::TFeatArray'),
        ('const TFeatArray&', 'const ncbi::objects::feature::CFeatTree::TFeatArray&'),

        (' CFeatInfo *', ' ncbi::objects::feature::CFeatTree::CFeatInfo *'),
        ('(CFeatInfo *)', '(ncbi::objects::feature::CFeatTree::CFeatInfo *)'),
        ('(eNone)', '(::eNone)'),

        ('ncbi::BASE64_Encode', 'BASE64_Encode'),
        ('ncbi::BASE64_Decode', 'BASE64_Decode'),

        (' SLink *', ' ncbi::cobalt::CLinks::SLink *'),
        ('(SLink const &)', '(ncbi::cobalt::CLinks::SLink const &)' ),

        (' CInputBioseq_CI *', ' ncbi::CWinMaskUtil::CInputBioseq_CI *'),
        ('(CInputBioseq_CI *)', '(ncbi::CWinMaskUtil::CInputBioseq_CI *)'),

        (' CWinMaskConfigException *', ' ncbi::CWinMaskConfig::CWinMaskConfigException *'),
        ('(CWinMaskConfigException const &)', '(ncbi::CWinMaskConfig::CWinMaskConfigException const &)'),

        (' TTemplateLibFilter *', ' ncbi::CHTMLPage::TTemplateLibFilter *'),
        ('(TTemplateLibFilter *)', '(ncbi::CHTMLPage::TTemplateLibFilter *)'),

        (' SMod *', ' ncbi::objects::CSourceModParser::SMod *'),
        ('(SMod const &)', '(ncbi::objects::CSourceModParser::SMod const &)'),

        (' EProperty ', ' ncbi::CSnpBitfield::EProperty '),
        (' EFunctionClass ', ' ncbi::CSnpBitfield::EFunctionClass '),
        (' IEncoding *', ' ncbi::CSnpBitfield::IEncoding *'),
        ('(IEncoding *)', '(ncbi::CSnpBitfield::IEncoding *)')

                   ]

    for lang in getTargetLanguages():
        src = lang + "_wrappers.cpp"
        dst = src + ".adj"

        fin = open( src )
        fout = open( dst, "w" )

        content = fin.read()
        for enum in enums:
            content = content.replace( 'enum ' + enum, enum )
        for item in replacements:
            content = content.replace( item[0], item[1] )

        fout.write( content )

        fin.close()
        fout.close()

        os.system( "rm -rf " + src )
        os.system( "mv " + dst + " " + src )

    return


def buildWrapperLibrary( buildDir, verbose ):
    """
    Replaces the fake 0-length source code and
    runs make in the proper directory
    """

    print "Building the wrapper library..."

    # For all target languages
    for lang in getTargetLanguages():
        srcFileName = lang + "_wrappers.cpp"
        targetCppFile = lang + "_wrappers/" + srcFileName
        print "Removing " + targetCppFile + "..."
        retCode = os.system( "rm -rf " + targetCppFile )
        retCode += os.system( "cp " + srcFileName + " " + targetCppFile )
        if retCode != 0:
            raise Exception( "Error replacing a dummy source code file " \
                             "with the generated one for " + lang +      \
                             ". Source file: " + srcFileName +           \
                             " Target file: " + targetCppFile )

        srcFileName = lang + "_wrappers.h"
        targetHeaderFile = lang + "_wrappers/" + srcFileName
        print "Removing " + targetHeaderFile + "..."
        os.system( "rm -rf " + targetHeaderFile )
        if os.path.exists( srcFileName ):
            if os.system( "cp " + srcFileName + " " + targetHeaderFile ) != 0:
                raise Exception( "Error replacing a dummy header file " \
                                 "with the generated one for " + lang + \
                                 ". Source file: " + srcFileName + \
                                 " Target file: " + targetHeaderFile )

        # Run make in the proper dir
        wrapperBuildDir = buildDir + "internal/cppcore/toolkit_wrappers/" + \
                          lang + "_wrappers"
        cmdLine = 'cd ' + wrapperBuildDir + '; make CXX_WRAPPER="" all_p'
        if verbose:
            print "Wrapper building command line: " + cmdLine

        if os.system( cmdLine ) != 0:
            raise Exception( "Error making wrapper for " + lang )

    return


def runSWIG( incDir, buildDir, verbose, collectOutput ):
    """
    Runs SWIG to generate the cpp wrapper file
    """

    # Pick up the -D keys which were used to build the toolkit libraries
    makefileInPath = buildDir + "Makefile.mk"
    if not os.path.exists( makefileInPath ):
        raise Exception( "Cannot find the build Makefile.mk. " \
                         "Expected it here: " + makefileInPath )

    defines = [ '-DNCBI_TEMPSTR_USE_A_COPY', '-DNCBI_SWIG' ]
    cppFlags = safeRun( [ "grep", "CONF_CPPFLAGS", makefileInPath ] ).strip()
    parts = cppFlags.split()
    for part in parts:
        part = part.strip()
        if part.startswith( "-D" ):
            defines.append( part )

    # Form the -I directives
    includes = []
    includes.append( "-I" + incDir[ :-1 ] )
    includes.append( "-I" + buildDir.replace( "/build/", "/inc" ) )
    includes.append( "-I/usr/local/gcc-4.0.1/include/g++-v3/" )
    # includes.append( "-I/usr/include" )

    # For all target languages
    for lang in getTargetLanguages():
        if verbose:
            print "Running SWIG for " + lang
        # w312 - nested struct
        # w314 - auto renaming is -> _is, in -> _in
        # w361 - operator! ignored
        # w362 - operator= ignored
        # w350 - operator new ignored
        # w394 - operator new[] ignored
        # w351 - operator delete ignored
        # w395 - operator delete[] ignored
        # w508 - Declaration of 'ZZZ' shadows declaration
        #        accessible via operator->()
        cmdLine  = "swig -w312,314,361,362,350,394,351,395,508 " \
                   "-v -c++ -importall -" + lang
        cmdLine += " " + " ".join( defines )
        cmdLine += " " + " ".join( includes )
        cmdLine += " -I./" + lang
        cmdLine += " -o " + lang + "_wrappers.cpp"
        cmdLine += " ncbi.i"
        cmdLine += " 2>&1 | grep -v -f swig_log_filter"

        if collectOutput:
            cmdLine += " | tee swig.log"

        if verbose:
            print "SWIG command line: " + cmdLine

        if os.system( cmdLine ) != 0:
            raise Exception( "Error running SWIG for " + lang )

        print "SWIG run completed for " + lang
        print "-----------------------------"

    return


def getTargetLanguages():
    """ Provides a list of the target languages """

    # default language is Python
    return os.environ.get( "NCBI_SWIG_LANG",
                           "python" ).strip().lower().split()


def toolkitHeaders( path, headers ):
    """ Builds a list of all the toolkit header files """

    for item in os.listdir( path ):
        dirItem = path + "/" + item
        if os.path.isdir( dirItem ):
            toolkitHeaders( dirItem, headers )
            continue
        if os.path.isfile( dirItem ):
            if dirItem.endswith( ".hpp" ) or dirItem.endswith( ".h" ):
                headers.append( dirItem )
    return


def parserError( parser, message ):
    """ Prints the message and help on stderr """

    sys.stdout = sys.stderr
    print message
    parser.print_help()
    return 2


def safeRun( commandArgs ):
    """
    Runs the given command and reads the output
    """

    errTmp = tempfile.mkstemp()
    errStream = os.fdopen( errTmp[0] )
    process = Popen( commandArgs, stdin = PIPE,
                     stdout = PIPE, stderr = errStream )
    process.stdin.close()
    processStdout = process.stdout.read()
    process.stdout.close()
    errStream.seek( 0 )
    err = errStream.read()
    errStream.close()
    os.unlink( errTmp[1] )
    process.wait()

    # 'grep' return codes:
    # 0 - OK, lines found
    # 1 - OK, no lines found
    # 2 - Error occured
    if process.returncode != 0 and commandArgs[0] != "grep":
        raise Exception( "Error in '%s' invocation: %s" % \
                         (commandArgs[0], err) )
    return processStdout


def getExceptionInfo():
    """
    The function formats the exception and returns the string which
    could be then printed or logged
    """

    excType, value, tback = sys.exc_info()
    msg = str( value )
    if len( msg ) == 0:
        msg = "There is no message associated with the exception."
    if msg.startswith( '(' ) and msg.endswith( ')' ):
        msg = msg[1:-1]

    try:
        tbInfo = format_tb( tback )
        tracebackInfoMsg = "Traceback information:\n" + "".join( tbInfo )
    except:
        tracebackInfoMsg = "No traceback information available"

    return "Exception is caught. " + msg + "\n" + tracebackInfoMsg



# The script execution entry point
if __name__ == "__main__":
    returnCode = 2
    try:
        returnCode = main()
    except:
        print >> sys.stderr, getExceptionInfo()

    if returnCode != 0:
        returnCode = 2
    sys.exit( returnCode )

