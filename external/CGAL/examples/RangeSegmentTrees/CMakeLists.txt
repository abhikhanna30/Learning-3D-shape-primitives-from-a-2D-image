# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


project( RangeSegmentTrees_example )

cmake_minimum_required(VERSION 2.8.10)

find_package(CGAL QUIET COMPONENTS Core )

if ( CGAL_FOUND )

  include( ${CGAL_USE_FILE} )

  include( CGAL_CreateSingleSourceCGALProgram )

  include_directories (BEFORE "../../include")

    include_directories (BEFORE "include")

  create_single_source_cgal_program( "range_tree_1.cpp" )
  create_single_source_cgal_program( "range_tree_2.cpp" )
  create_single_source_cgal_program( "range_tree_3.cpp" )
  create_single_source_cgal_program( "range_tree_4.cpp" )
  create_single_source_cgal_program( "range_tree_map_2.cpp" )
  create_single_source_cgal_program( "range_tree_set_2.cpp" )
  create_single_source_cgal_program( "segment_tree_1.cpp" )
  create_single_source_cgal_program( "segment_tree_2.cpp" )
  create_single_source_cgal_program( "segment_tree_3.cpp" )
  create_single_source_cgal_program( "segment_tree_4.cpp" )
  create_single_source_cgal_program( "segment_tree_map_2.cpp" )
  create_single_source_cgal_program( "segment_tree_set_2.cpp" )
  create_single_source_cgal_program( "segment_tree_set_3.cpp" )
  create_single_source_cgal_program( "software_design_rgs.cpp" )

else()
  
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
  
endif()

