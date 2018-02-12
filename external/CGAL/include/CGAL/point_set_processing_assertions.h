// Copyright (c) 2007  INRIA (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
// You can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id$
// 
//
// Author(s)     : Geert-Jan Giezeman, Sven Schoenherr, Laurent Saboret
//
// Generated from script create_assertions.sh


/// \cond SKIP_IN_MANUAL

/// @file point_set_processing_assertions.h
/// Define checking macros for the Point_set_processing_3 package

// Note that this header file is intentionnaly not protected with a
// macro (as <cassert>). Calling it a second time with another value
// for NDEBUG for example must make a difference.

#include <CGAL/assertions.h>

// macro definitions
// =================
// assertions
// ----------

#undef CGAL_point_set_processing_assertion
#undef CGAL_point_set_processing_assertion_msg
#undef CGAL_point_set_processing_assertion_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS) || defined(CGAL_NO_ASSERTIONS) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_assertion(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_assertion_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_assertion_code(CODE)
#else
#  define CGAL_point_set_processing_assertion(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_assertion_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_assertion_code(CODE) CODE
#  define CGAL_point_set_processing_assertions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS


#undef CGAL_point_set_processing_exactness_assertion
#undef CGAL_point_set_processing_exactness_assertion_msg
#undef CGAL_point_set_processing_exactness_assertion_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS) || defined(CGAL_NO_ASSERTIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || defined(NDEBUG)
#  define CGAL_point_set_processing_exactness_assertion(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_assertion_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_assertion_code(CODE)
#else
#  define CGAL_point_set_processing_exactness_assertion(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_exactness_assertion_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_exactness_assertion_code(CODE) CODE
#  define CGAL_point_set_processing_exactness_assertions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS


#undef CGAL_point_set_processing_expensive_assertion
#undef CGAL_point_set_processing_expensive_assertion_msg
#undef CGAL_point_set_processing_expensive_assertion_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS) \
  || defined(CGAL_NO_ASSERTIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_assertion(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_assertion_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_assertion_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_assertion(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_assertion_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_assertion_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_assertions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS


#undef CGAL_point_set_processing_expensive_exactness_assertion
#undef CGAL_point_set_processing_expensive_exactness_assertion_msg
#undef CGAL_point_set_processing_expensive_exactness_assertion_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS) || defined(CGAL_NO_ASSERTIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_exactness_assertion(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_assertion_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_assertion_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_exactness_assertion(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_exactness_assertion_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::assertion_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_exactness_assertion_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_exactness_assertions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_ASSERTIONS


// preconditions
// -------------

#undef CGAL_point_set_processing_precondition
#undef CGAL_point_set_processing_precondition_msg
#undef CGAL_point_set_processing_precondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS) || defined(CGAL_NO_PRECONDITIONS) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_precondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_precondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_precondition_code(CODE)
#else
#  define CGAL_point_set_processing_precondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_precondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_precondition_code(CODE) CODE
#  define CGAL_point_set_processing_preconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS


#undef CGAL_point_set_processing_exactness_precondition
#undef CGAL_point_set_processing_exactness_precondition_msg
#undef CGAL_point_set_processing_exactness_precondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS) || defined(CGAL_NO_PRECONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || defined(NDEBUG)
#  define CGAL_point_set_processing_exactness_precondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_precondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_precondition_code(CODE)
#else
#  define CGAL_point_set_processing_exactness_precondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_exactness_precondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_exactness_precondition_code(CODE) CODE
#  define CGAL_point_set_processing_exactness_preconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS


#undef CGAL_point_set_processing_expensive_precondition
#undef CGAL_point_set_processing_expensive_precondition_msg
#undef CGAL_point_set_processing_expensive_precondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS) || defined(CGAL_NO_PRECONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_precondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_precondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_precondition_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_precondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_precondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_precondition_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_preconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS


#undef CGAL_point_set_processing_expensive_exactness_precondition
#undef CGAL_point_set_processing_expensive_exactness_precondition_msg
#undef CGAL_point_set_processing_expensive_exactness_precondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS) || defined(CGAL_NO_PRECONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_exactness_precondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_precondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_precondition_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_exactness_precondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_exactness_precondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::precondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_exactness_precondition_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_exactness_preconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_PRECONDITIONS


// postconditions
// --------------

#undef CGAL_point_set_processing_postcondition
#undef CGAL_point_set_processing_postcondition_msg
#undef CGAL_point_set_processing_postcondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS) || defined(CGAL_NO_POSTCONDITIONS) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_postcondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_postcondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_postcondition_code(CODE)
#else
#  define CGAL_point_set_processing_postcondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_postcondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_postcondition_code(CODE) CODE
#  define CGAL_point_set_processing_postconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS


#undef CGAL_point_set_processing_exactness_postcondition
#undef CGAL_point_set_processing_exactness_postcondition_msg
#undef CGAL_point_set_processing_exactness_postcondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS) || defined(CGAL_NO_POSTCONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || defined(NDEBUG)
#  define CGAL_point_set_processing_exactness_postcondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_postcondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_postcondition_code(CODE)
#else
#  define CGAL_point_set_processing_exactness_postcondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_exactness_postcondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_exactness_postcondition_code(CODE) CODE
#  define CGAL_point_set_processing_exactness_postconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS


#undef CGAL_point_set_processing_expensive_postcondition
#undef CGAL_point_set_processing_expensive_postcondition_msg
#undef CGAL_point_set_processing_expensive_postcondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS) || defined(CGAL_NO_POSTCONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_postcondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_postcondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_postcondition_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_postcondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_postcondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_postcondition_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_postconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS


#undef CGAL_point_set_processing_expensive_exactness_postcondition
#undef CGAL_point_set_processing_expensive_exactness_postcondition_msg
#undef CGAL_point_set_processing_expensive_exactness_postcondition_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS) || defined(CGAL_NO_POSTCONDITIONS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_exactness_postcondition(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_postcondition_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_postcondition_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_exactness_postcondition(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_exactness_postcondition_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::postcondition_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_exactness_postcondition_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_exactness_postconditions 1
#endif // CGAL_POINT_SET_PROCESSING_NO_POSTCONDITIONS


// warnings
// --------

#undef CGAL_point_set_processing_warning
#undef CGAL_point_set_processing_warning_msg
#undef CGAL_point_set_processing_warning_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_WARNINGS) || defined(CGAL_NO_WARNINGS) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_warning(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_warning_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_warning_code(CODE)
#else
#  define CGAL_point_set_processing_warning(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_warning_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_warning_code(CODE) CODE
#  define CGAL_point_set_processing_warnings 1
#endif // CGAL_POINT_SET_PROCESSING_NO_WARNINGS


#undef CGAL_point_set_processing_exactness_warning
#undef CGAL_point_set_processing_exactness_warning_msg
#undef CGAL_point_set_processing_exactness_warning_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_WARNINGS) || defined(CGAL_NO_WARNINGS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || defined(NDEBUG)
#  define CGAL_point_set_processing_exactness_warning(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_warning_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_exactness_warning_code(CODE)
#else
#  define CGAL_point_set_processing_exactness_warning(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_exactness_warning_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_exactness_warning_code(CODE) CODE
#  define CGAL_point_set_processing_exactness_warnings 1
#endif // CGAL_POINT_SET_PROCESSING_NO_WARNINGS


#undef CGAL_point_set_processing_expensive_warning
#undef CGAL_point_set_processing_expensive_warning_msg
#undef CGAL_point_set_processing_expensive_warning_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_WARNINGS) || defined(CGAL_NO_WARNINGS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_warning(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_warning_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_warning_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_warning(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_warning_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_warning_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_warnings 1
#endif // CGAL_POINT_SET_PROCESSING_NO_WARNINGS


#undef CGAL_point_set_processing_expensive_exactness_warning
#undef CGAL_point_set_processing_expensive_exactness_warning_msg
#undef CGAL_point_set_processing_expensive_exactness_warning_code

#if defined(CGAL_POINT_SET_PROCESSING_NO_WARNINGS) || defined(CGAL_NO_WARNINGS) \
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXACTNESS) && !defined(CGAL_CHECK_EXACTNESS))\
  || (!defined(CGAL_POINT_SET_PROCESSING_CHECK_EXPENSIVE) && !defined(CGAL_CHECK_EXPENSIVE)) \
  || defined(NDEBUG)
#  define CGAL_point_set_processing_expensive_exactness_warning(EX) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_warning_msg(EX,MSG) (static_cast<void>(0))
#  define CGAL_point_set_processing_expensive_exactness_warning_code(CODE)
#else
#  define CGAL_point_set_processing_expensive_exactness_warning(EX) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__))
#  define CGAL_point_set_processing_expensive_exactness_warning_msg(EX,MSG) \
   (CGAL::possibly(EX)?(static_cast<void>(0)): ::CGAL::warning_fail( # EX , __FILE__, __LINE__, MSG))
#  define CGAL_point_set_processing_expensive_exactness_warning_code(CODE) CODE
#  define CGAL_point_set_processing_expensive_exactness_warnings 1
#endif // CGAL_POINT_SET_PROCESSING_NO_WARNINGS

/// \endcond
