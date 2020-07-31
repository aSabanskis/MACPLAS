#!/bin/bash

#
# Formats the source code according to .clang-format
#

# Provide full path to clang-format-6.0 or add its directory to PATH.
# Versions below 6 are not supported.
formatter=clang-format

find include tests \
-regextype egrep -regex ".*\.(cc|h)" -print0 | \
xargs -0 -n 1 -I {} bash -c "echo {}; $formatter -i {}"
