#!/bin/sh

INSTALL_DIR=$1
VERSION="${NCBI_PACKAGE_VERSION}"

if [ -z "$INSTALL_DIR" ]; then
    echo "Installation directory is empty"
    exit 1
fi
TARBALL_DIR="$INSTALL_DIR/build_tarball"

PYTHON_ROOT=/opt/python-2.5

if [ ! -d "$PYTHON_ROOT" ]; then
    echo "Cannot find Python directory"
    exit 2
fi

PY='from distutils import sysconfig; print sysconfig.get_config_var("LIBDEST")'
PYTHON_DIR="`$PYTHON_ROOT/bin/python -c \"$PY\"`/site-packages"
PYTHON_DEST="$TARBALL_DIR/$PYTHON_DIR"

mkdir -p "$PYTHON_DEST" || exit 3
install -m 644 $INSTALL_DIR/lib/_ncbi_lb.so $INSTALL_DIR/lib/ncbi_lb.py \
    "$PYTHON_DEST" || exit 4

PERL_ROOT=/opt/perl-5.8.8

if [ ! -d "$PERL_ROOT" ]; then
    echo "Cannot find Perl directory"
    exit 5
fi

PERL_DIR="`$PERL_ROOT/bin/perl -MConfig -e 'print $Config{vendorarchexp}'`"
PERL_DEST=$TARBALL_DIR/$PERL_DIR

mkdir -p $PERL_DEST/auto/ncbi_lb || exit 5
install -m 644 $INSTALL_DIR/lib/ncbi_lb.pm $PERL_DEST || exit 6
install -m 644 $INSTALL_DIR/lib/ncbi_lb.so $PERL_DEST/auto/ncbi_lb || exit 7

tar -czf "$INSTALL_DIR/installation.tar.gz" -C "$TARBALL_DIR" opt || exit 8

rm -rf "$TARBALL_DIR"
