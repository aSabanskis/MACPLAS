#!/bin/bash

#
# Formats the source code according to .clang-format
#

# Provide full path to clang-format or add its directory to PATH.
# Versions below 11 are not supported.
formatter=clang-format-11

find include tests applications \
-regextype egrep -regex ".*\.(cc|h)" -print0 | \
xargs -0 -n 1 -I {} bash -c "echo {}; $formatter -i {}"
