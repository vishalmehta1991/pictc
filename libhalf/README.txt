HALF-PRECISION FLOATING POINT LIBRARY (Version 2.0.0)
------------------------------------------------------

This is a C++ header-only library to provide an IEEE 754 conformant 16-bit 
half-precision floating point type along with corresponding arithmetic 
operators, type conversions and common mathematical functions. It aims for both 
efficiency and ease of use, trying to accurately mimic the behaviour of the 
built-in floating point types at the best performance possible.


INSTALLATION AND REQUIREMENTS
-----------------------------

Conveniently, the library consists of just a single header file containing all 
the functionality, which can be directly included by your projects, without the 
neccessity to build anything or link to anything.

Whereas this library is fully C++98-compatible, it can profit from certain 
C++11 features. Support for those features is checked automatically at compile 
(or rather preprocessing) time, but can be explicitly enabled or disabled by 
defining the corresponding preprocessor symbols to either 1 or 0 yourself. This 
is useful when the automatic detection fails (for more exotic implementations) 
or when a feature should be explicitly disabled:

  - 'long long' integer type for mathematical functions returning 'long long' 
    results (enabled for VC++ 2003 and newer, gcc and clang, overridable with 
    'HALF_ENABLE_CPP11_LONG_LONG').

  - Static assertions for extended compile-time checks (enabled for VC++ 2010, 
    gcc 4.3, clang 2.9 and newer, overridable with 'HALF_ENABLE_CPP11_STATIC_ASSERT').

  - Generalized constant expressions (enabled for VC++ 2015, gcc 4.6, clang 3.1 
    and newer, overridable with 'HALF_ENABLE_CPP11_CONSTEXPR').

  - noexcept exception specifications (enabled for VC++ 2015, gcc 4.6, clang 3.0 
    and newer, overridable with 'HALF_ENABLE_CPP11_NOEXCEPT').

  - User-defined literals for half-precision literals to work (enabled for 
    VC++ 2015, gcc 4.7, clang 3.1 and newer, overridable with 
    'HALF_ENABLE_CPP11_USER_LITERALS').

  - Type traits and template meta-programming features from <type_traits> 
    (enabled for VC++ 2010, libstdc++ 4.3, libc++ and newer, overridable with 
    'HALF_ENABLE_CPP11_TYPE_TRAITS').

  - Special integer types from <cstdint> (enabled for VC++ 2010, libstdc++ 4.3, 
    libc++ and newer, overridable with 'HALF_ENABLE_CPP11_CSTDINT').

  - Certain C++11 single-precision mathematical functions from <cmath> for 
    floating point classification during conversions from higher precision types 
    (enabled for VC++ 2013, libstdc++ 4.3, libc++ and newer, overridable with 
    'HALF_ENABLE_CPP11_CMATH').

  - Hash functor 'std::hash' from <functional> (enabled for VC++ 2010, 
    libstdc++ 4.3, libc++ and newer, overridable with 'HALF_ENABLE_CPP11_HASH').

The library has been tested successfully with Visual C++ 2005-2015, gcc 4-8 
and clang 3-8 on 32- and 64-bit x86 systems. Please contact me if you have any 
problems, suggestions or even just success testing it on other platforms.


DOCUMENTATION
-------------

What follows are some general words about the usage of the library and its 
implementation. For a complete documentation of its interface consult the 
corresponding website http://half.sourceforge.net. You may also generate the 
complete developer documentation from the library's only include file's doxygen 
comments, but this is more relevant to developers rather than mere users.

BASIC USAGE

To make use of the library just include its only header file half.hpp, which 
defines all half-precision functionality inside the 'half_float' namespace. The 
actual 16-bit half-precision data type is represented by the 'half' type. This 
type behaves like the builtin floating point types as much as possible, 
supporting the usual arithmetic, comparison and streaming operators, which 
makes its use pretty straight-forward:

    using half_float::half;
    half a(3.4), b(5);
    half c = a * b;
    c += 3;
    if(c > a)
        std::cout << c << std::endl;

Additionally the 'half_float' namespace also defines half-precision versions 
for all mathematical functions of the C++ standard library, which can be used 
directly through ADL:

    half a(-3.14159);
    half s = sin(abs(a));
    long l = lround(s);

You may also specify explicit half-precision literals, since the library 
provides a user-defined literal inside the 'half_float::literal' namespace, 
which you just need to import (assuming support for C++11 user-defined literals):

    using namespace half_float::literal;
    half x = 1.0_h;

Furthermore the library provides proper specializations for 
'std::numeric_limits', defining various implementation properties, and 
'std::hash' for hashing half-precision numbers (assuming support for C++11 
'std::hash'). Similar to the corresponding preprocessor symbols from <cmath> 
the library also defines the 'HUGE_VALH' constant and maybe the 'FP_FAST_FMAH' 
symbol.

CONVERSIONS AND ROUNDING

The half is explicitly constructible/convertible from a single-precision float 
argument. Thus it is also explicitly constructible/convertible from any type 
implicitly convertible to float, but constructing it from types like double or 
int will involve the usual warnings arising when implicitly converting those to 
float because of the lost precision. On the one hand those warnings are 
intentional, because converting those types to half neccessarily also reduces 
precision. But on the other hand they are raised for explicit conversions from 
those types, when the user knows what he is doing. So if those warnings keep 
bugging you, then you won't get around first explicitly converting to float 
before converting to half, or use the 'half_cast' described below. In addition 
you can also directly assign float values to halfs.

In contrast to the float-to-half conversion, which reduces precision, the 
conversion from half to float (and thus to any other type implicitly 
convertible from float) is implicit, because all values represetable with 
half-precision are also representable with single-precision. This way the 
half-to-float conversion behaves similar to the builtin float-to-double 
conversion and all arithmetic expressions involving both half-precision and 
single-precision arguments will be of single-precision type. This way you can 
also directly use the mathematical functions of the C++ standard library, 
though in this case you will invoke the single-precision versions which will 
also return single-precision values, which is (even if maybe performing the 
exact same computation, see below) not as conceptually clean when working in a 
half-precision environment.

The default rounding mode for conversions between half and more precise types 
as well as for rounding results of arithmetic operations and mathematical 
functions rounds to the nearest representable value. But by redefining the 
'HALF_ROUND_STYLE' preprocessor symbol (before including half.hpp) this default 
can be overridden with one of the other standard rounding modes using their 
respective constants or the equivalent values of 'std::float_round_style' (it 
can even be synchronized with the built-in single-precision implementation by 
defining it to 'std::numeric_limits<float>::round_style'):

  - 'std::round_indeterminate' (-1) for the fastest rounding.

  - 'std::round_toward_zero' (0) for rounding toward zero.

  - std::round_to_nearest' (1) for rounding to the nearest value (default).

  - std::round_toward_infinity' (2) for rounding toward positive infinity.

  - std::round_toward_neg_infinity' (3) for rounding toward negative infinity.

In addition to changing the overall default rounding mode one can also use the 
'half_cast'. This converts between half and any built-in arithmetic type using 
a configurable rounding mode (or the default rounding mode if none is 
specified). In addition to a configurable rounding mode, 'half_cast' has 
another big difference to a mere 'static_cast': Any conversions are performed 
directly using the given rounding mode, without any intermediate conversion 
to/from 'float'. This is especially relevant for conversions to integer types, 
which don't necessarily truncate anymore. But also for conversions from 
'double' or 'long double' this may produce more precise results than a 
pre-conversion to 'float' using the single-precision implementation's current 
rounding mode would.

    half a = half_cast<half>(4.2);
    half b = half_cast<half,std::numeric_limits<float>::round_style>(4.2f);
    assert( half_cast<int, std::round_to_nearest>( 0.7_h )     == 1 );
    assert( half_cast<half,std::round_toward_zero>( 4097 )     == 4096.0_h );
    assert( half_cast<half,std::round_toward_infinity>( 4097 ) == 4100.0_h );
    assert( half_cast<half,std::round_toward_infinity>( std::numeric_limits<double>::min() ) > 0.0_h );

ACCURACY AND PERFORMANCE

From version 2.0 onward the library is implemented without employing the 
underlying floating point implementation of the system (except for conversions, 
of course), providing an entirely self-contained half-precision implementation 
with results independent from the system's existing single- or double-precision 
implementation and its rounding behaviour.

As to accuracy, many of the operators and functions provided by this library 
are exact to rounding for all rounding modes, i.e. the error to the exact 
result is at most 0.5 ULP (unit in the last place) for rounding to nearest and 
less than 1 ULP for all other rounding modes. This holds for all the operations 
required by the IEEE 754 standard and many more. Specifically the following 
functions might exhibit a deviation from the correctly rounded result by 1 ULP 
for a select few input values: 'expm1', 'log1p', 'pow', 'atan2', 'erf', 'erfc', 
'lgamma', 'tgamma' (for more details see the documentation of the individual 
functions). All other functions and operators are always exact to rounding or 
independent of the rounding mode altogether.

The increased IEEE-conformance and cleanliness of this implementation comes 
with a certain performance cost compared to doing computations and mathematical 
functions in hardware-accelerated single-precision. On average and depending on 
the platform, the arithemtic operators are about 75% as fast and the 
mathematical functions about 33-50% as fast as performing the corresponding 
operations in single-precision and converting between the inputs and outputs. 
However, directly computing with half-precision values is a rather rare 
use-case and usually using actual 'float' values for all computations and 
temproraries and using 'half's only for storage is the recommended way. But 
nevertheless the goal of this library was to provide a complete and 
conceptually clean IEEE-confromant half-precision implementation and in the few 
cases when you do need to compute directly in half-precision you do so for a 
reason and want accurate results.

If necessary, this internal implementation can be overridden by defining the 
'HALF_ARITHMETIC_TYPE' preprocessor symbol (before including half.hpp) to one 
of the built-in floating point types ('float', 'double' or 'long double'), 
which will cause the library to use this type for computing arithmetic 
operations and mathematical functions (if available). However, due to using the 
platform's floating point implementation (and its rounding behaviour) 
internally, this might cause results to deviate from the specified 
half-precision rounding mode.

The conversion operations between half-precision and single-precision types can 
also make use of the F16C extension for x86 processors by using the 
corresponding compiler intrinsics from <immintrin.h>. Support for this is 
checked at compile-time by looking for the '__F16C__' macro which at least gcc 
and clang define based on the target platform. It can also be enabled manually 
by defining the 'HALF_ENABLE_F16C_INTRINSICS' preprocessor symbol to 1 (before 
including half.hpp), or 0 for explicitly disabling it. However, this will 
directly use the corresponding intrinsics for conversion without checking if 
they are available at runtime (possibly crashing if they are not), so make sure 
they are supported on the target platform before enabling this.

IEEE CONFORMANCE

The half type uses the standard IEEE representation with 1 sign bit, 5 exponent 
bits and 10 mantissa bits (11 when counting the hidden bit). It supports all 
types of special values, like subnormal values, infinity and NaNs, specifically 
in the area of floating point exceptions and signaling NaNs. The library does 
not differentiate between signaling and quiet NaNs, this means operations on 
halfs are not specified to trap on signaling NaNs. The library also does not 
provide any floating point exceptions, thus arithmetic operations or 
mathematical functions are not specified to invoke proper floating point 
exceptions.

Some of those points could have been circumvented by controlling the floating 
point environment using <cfenv> or implementing a similar exception mechanism. 
But this would have required excessive runtime checks giving two high an impact 
on performance for something that is rarely ever needed, so it was reserved for 
possible future versions. If you really need to rely on proper floating point 
exceptions, it is recommended to explicitly perform computations using the 
built-in floating point types to be on the safe side.


CREDITS AND CONTACT
-------------------

This library is developed by CHRISTIAN RAU and released under the MIT License 
(see LICENSE.txt). If you have any questions or problems with it, feel free to 
contact me at rauy@users.sourceforge.net.

Additional credit goes to JEROEN VAN DER ZIJP for his paper on "Fast Half Float 
Conversions", whose algorithms have been used in the library for converting 
between half-precision and single-precision values.
