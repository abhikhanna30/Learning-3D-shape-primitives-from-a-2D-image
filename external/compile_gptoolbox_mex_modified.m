path_to_eltopo = './';
path_to_libigl = '../../libigl/';
verbose = true;
CGAL='../../CGAL/';
LOCAL_INC = '-I$HOME/local/include';
LOCAL_LIB = '-L$HOME/local/lib';
MEXOPTS={'-largeArrayDims','-DMEX'};
if verbose
  MEXOPTS={'-v',MEXOPTS{:}};
end
MSSE42='CXXFLAGS=$CXXFLAGS -msse4.2';
STDCPP11='CXXFLAGS=$CXXFLAGS -std=c++11';

ELTOPO_INC= sprintf('-I%s/',path_to_eltopo);
ELTOPO_LIB= strsplit(sprintf('-L%s/eltopo3d -leltopo_release',path_to_eltopo));
CLANG={'CXX=/usr/bin/clang++','LD=/usr/bin/clang++'};
FRAMEWORK_LDFLAGS='LDFLAGS=\$LDFLAGS -framework Foundation -framework AppKit -framework Accelerate';
NOOPT_LDOPTIMFLAGS='LDOPTIMFLAGS="-O "';

% See libigl documentation. In short, Libigl is a header-only library by
% default: no compilation needed (like Eigen). There's an advanced **option**
% to precompile libigl as a static library. This cuts down on compilation time.
% It is optional and more difficult to set up. Set this to true only if you
% know what you're doing.
%use_libigl_static_library  = false;
use_libigl_static_library = exist([path_to_libigl '/lib/libigl.a'],'file')~=0;
if use_libigl_static_library
  r = -1;
  if strcmp(char(java.lang.System.getProperty('user.name')),'ajx')
    cmd = sprintf('cd "%s"/lib && /usr/local/bin/cmake ../optional/ -DLIBIGL_USE_STATIC_LIBRARY=ON -DLIBIGL_WITH_ANTTWEAKBAR=ON -DLIBIGL_WITH_BBW=ON -DLIBIGL_WITH_CORK=ON -DLIBIGL_WITH_CGAL=ON -DLIBIGL_WITH_COMISO=ON -DLIBIGL_WITH_EMBREE=ON -DLIBIGL_WITH_LIM=ON -DLIBIGL_WITH_MATLAB=ON -DLIBIGL_WITH_MOSEK=ON -DLIBIGL_WITH_NANOGUI=OFF -DLIBIGL_WITH_OPENGL=ON -DLIBIGL_WITH_PNG=ON -DLIBIGL_WITH_TETGEN=ON -DLIBIGL_WITH_TRIANGLE=ON -DLIBIGL_WITH_VIEWER=ON -DLIBIGL_WITH_XML=ON -DCMAKE_BUILD_TYPE=Release && ../scripts/make.sh -j3',path_to_libigl);
  else
    cmd = sprintf('cd "%s"/lib && /usr/local/bin/cmake ../optional/ -DLIBIGL_USE_STATIC_LIBRARY=ON -DLIBIGL_WITH_ANTTWEAKBAR=ON -DLIBIGL_WITH_BBW=ON -DLIBIGL_WITH_CORK=ON -DLIBIGL_WITH_CGAL=ON -DLIBIGL_WITH_COMISO=ON -DLIBIGL_WITH_EMBREE=ON -DLIBIGL_WITH_LIM=ON -DLIBIGL_WITH_MATLAB=ON -DLIBIGL_WITH_MOSEK=ON -DLIBIGL_WITH_NANOGUI=OFF -DLIBIGL_WITH_OPENGL=ON -DLIBIGL_WITH_PNG=ON -DLIBIGL_WITH_TETGEN=ON -DLIBIGL_WITH_TRIANGLE=ON -DLIBIGL_WITH_VIEWER=ON -DLIBIGL_WITH_XML=ON -DCMAKE_BUILD_TYPE=Release && make -j3',path_to_libigl);
  end
  if verbose
    fprintf('%s\n',cmd);
  end
  [r,c] = system(cmd);
  if r ~= 0
    warning('libigl Make error: (c set to command output)');
    use_libigl_static_library = false;
  end
end
%use_libigl_static_library= false;
LIBIGL_INC=sprintf('-I%s/include',path_to_libigl);
if use_libigl_static_library
  LIBIGL_FLAGS='-DIGL_STATIC_LIBRARY';
  LIBIGL_LIB=strsplit(sprintf('-L%s/lib -ligl',path_to_libigl));
  LIBIGL_LIBEMBREE='-liglembree';
  LIBIGL_LIBMATLAB='-liglmatlab';
  LIBIGL_LIBCGAL='-liglcgal';
  LIBIGL_LIBCORK='-liglcork';
else
  % `mex` has a silly requirement that arguments be non-empty, hence the NOOP
  % defines
  LIBIGL_FLAGS='-DIGL_SKIP';
  LIBIGL_LIB={'-DIGL_SKIP'};
  LIBIGL_LIBMATLAB='-DIGL_SKIP';
  LIBIGL_LIBEMBREE='-DIGL_SKIP';
  LIBIGL_LIBCGAL='-DIGL_SKIP';
  LIBIGL_LIBCORK='-DIGL_SKIP';
end
LIBIGL_BASE={LIBIGL_INC,LIBIGL_FLAGS,LIBIGL_LIB{:},LIBIGL_LIBMATLAB};

if exist(sprintf('%s/external/nanogui/ext/eigen/',path_to_libigl))
  EIGEN_INC=sprintf('-I%s/external/nanogui/ext/eigen/',path_to_libigl);
elseif exist('/usr/local/include/eigen3')
  EIGEN_INC='-I/usr/local/include/eigen3';
elseif exist('/opt/local/include/eigen3')
  EIGEN_INC='-I/opt/local/include/eigen3';
end

EMBREE=[path_to_libigl '/external/embree'];
EMBREE_INC=strsplit(sprintf('-I%s -I%s/include/',EMBREE,EMBREE));
EMBREE_LIB=strsplit(sprintf('-L%s/build -lembree -lsys -lembree_avx -lembree_avx2 -lembree_sse42 -llexers -lsimd',EMBREE));

CORK_INC=sprintf('-I%s/src',[path_to_libigl '/external/cork']);
CORK_LIB=strsplit(sprintf('-L%s -lcork',[path_to_libigl '/lib']));

TINYXML2=[path_to_libigl '/external/tinyxml2'];
TINYXML2_INC=sprintf('-I%s/',TINYXML2);
TINYXML2_LIB=strsplit(sprintf('-L%s/ -ltinyxml2',TINYXML2));

if exist('/usr/local/include/CGAL')
  CGAL='/usr/local/';
elseif exist('/opt/local/include/CGAL')
  CGAL='/opt/local/';
end
CGAL_INC=sprintf('-I%s/include',CGAL);
CGAL_LIB=strsplit(sprintf('-L%s/lib -lCGAL -lCGAL_Core -lgmp -lmpfr',CGAL));
CGAL_FLAGS='CXXFLAGS=\$CXXFLAGS -frounding-math';

BOOST='/usr/local/';
BOOST_INC=sprintf('-I%s/include',BOOST);
BOOST_LIB=strsplit(sprintf('-L%s/lib -lboost_thread-mt -lboost_system-mt',BOOST));

warning([ ...
  'This is **VERY** experimental. In principle, this should compile all ', ...
  'of the mex functions in this directory. In practice, users will ', ...
  'surely have to adjust paths and flags at the top of this file.']);
input('Hit any key to continue...');

out_of_date = @(prefix) ...
  ~exist(['./' prefix '.mexmaci64'],'file') || ...
  (datenum(getfield(dir([prefix '.cpp']),'date')) > ...
  datenum(getfield(dir([prefix '.mexmaci64']),'date'))) ||  ...
  (use_libigl_static_library  &&  ...
  datenum(getfield(dir([path_to_libigl '/lib/libigl.a']),'date')) > ...
  datenum(getfield(dir([prefix '.mexmaci64']),'date')));

prefixes = { ...
  'point_mesh_squared_distance', ...
};

for prefix = prefixes
  prefix = prefix{1};
  if out_of_date(prefix)
    try
      mex( ...
         MEXOPTS{:},...
         LOCAL_INC, ...
         LOCAL_LIB, ...
         MSSE42, ...
         STDCPP11, ...
         EIGEN_INC, ...
         ELTOPO_INC, ...
         LIBIGL_BASE{:}, ...
         LIBIGL_LIBEMBREE, ...
         LIBIGL_LIBCGAL, ...
         LIBIGL_LIBCORK, ...
         CGAL_INC,CGAL_LIB{:},CGAL_FLAGS, ...
        [prefix '.cpp']);
     catch ME
       warning(ME.identifier,ME.message);
     end
  end
end
