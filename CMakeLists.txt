# Copyright (C) 2026 Kent Gorday
# SPDX-License-Identifier: LGPL-3.0-only
#
# Builds psi4_swig_pcm._gcharge through Psi4's plugin interface (share/cmake/psi4):
# add_psi4_plugin links psi4::core, OpenMP and C++20, and on macOS leaves Psi4's
# symbols to be resolved from psi4.core at import (-undefined dynamic_lookup).
# The psi4PluginCache.cmake from `psi4 --plugin-compile` is not used: it sets
# -march=native, which would tie the binary to the build machine.

cmake_minimum_required(VERSION 3.19...4.2)
project(psi4_swig_pcm LANGUAGES CXX)

# psi4Config.cmake loads libint2-config.cmake, which asks for Boost::headers
# (Boost.Preprocessor, used inside libint2's own headers).  None of the headers
# this extension includes reach Boost or libint2, so Boost is needed only to get
# through configuration.  conda-forge's libboost-headers carries no CMake config,
# and libboost-devel would add libboost to the package's run dependencies, so
# the header-only target is declared here; libint2-config.cmake checks for it.
if(NOT TARGET Boost::headers)
  add_library(Boost::headers INTERFACE IMPORTED)
endif()

find_package(psi4 1.11 CONFIG REQUIRED)

add_psi4_plugin(_gcharge csrc/gcharge.cpp)
set_target_properties(_gcharge PROPERTIES SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}")

# The nonelectrostatic correction's field integrals call libint2's C++ engine directly
# (Psi4's own PotentialInt stops at first derivatives), so this one does include libint2,
# and through it Eigen and the Boost.Preprocessor headers.
add_psi4_plugin(_nonelectrostatic csrc/nonelectrostatic.cpp)
target_link_libraries(_nonelectrostatic PRIVATE Libint2::int-cxx-headeronly-shared)
set_target_properties(_nonelectrostatic PROPERTIES SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}")

install(TARGETS _gcharge _nonelectrostatic LIBRARY DESTINATION psi4_swig_pcm)

# The Psi4 version this binary was compiled against, checked at import.
configure_file(csrc/_build_info.py.in _build_info.py @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_build_info.py DESTINATION psi4_swig_pcm)
