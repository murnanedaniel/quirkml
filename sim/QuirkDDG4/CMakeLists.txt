#==========================================================================
#  QuirkDDG4 : a DDG4 plugin that runs the ATLAS infracolor-string quirk
#  generator (HyperbolaStepper / QuirkTransportation) inside ddsim / DDG4.
#
#  Build (with the ColliderML key4hep environment sourced -- see build.sh):
#    cmake -S . -B build -DDD4hep_DIR=$DD4hepINSTALL
#    cmake --build build -j8
#  then put build/ on DD4HEP_LIBRARY_PATH so ddsim discovers the factory.
#==========================================================================
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(QuirkDDG4 LANGUAGES CXX)

IF(NOT TARGET DD4hep::DDCore)
  find_package(DD4hep REQUIRED COMPONENTS DDG4)
  include(${DD4hep_DIR}/cmake/DD4hep.cmake)
  include(${DD4hep_DIR}/cmake/DD4hepBuild.cmake)
  dd4hep_configure_output()
ENDIF()

dd4hep_set_compiler_flags()

if(NOT TARGET Geant4::Interface)
  message(FATAL_ERROR "QuirkDDG4 requires Geant4 (DD4HEP_USE_GEANT4)")
endif()

# Geant4 is built multithreaded; its headers (G4AutoLock -> std::mutex) require
# -pthread, which is otherwise not propagated to this plugin target.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#---  the plugin library (factory auto-registered via the generated .components) ---
dd4hep_add_plugin(QuirkDDG4
  SOURCES src/*.cpp
  USES    DD4hep::DDCore DD4hep::DDG4 Geant4::Interface ROOT::Core
  )

# the ported Athena sources keep their headers alongside the .cpp in src/
target_include_directories(QuirkDDG4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(QuirkDDG4 PRIVATE Threads::Threads)
target_compile_options(QuirkDDG4 PRIVATE -pthread)

install(TARGETS QuirkDDG4 DESTINATION lib)
