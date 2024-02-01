#include <signal.h>
#include <cstdio>
#include <unistd.h>
#include <mutex>
#include <execinfo.h>
#include <csignal>
#include <sys/wait.h>
#include <cerrno>
#include <iostream>
#include <string.h>
#include <stdarg.h>
// more info: -g, gdb
// /etc/sysctl.d/10-ptrace.conf : change kernel.yama.ptrace_scope to 0
// may be debug in root mode

std::mutex crash_lock_;
/**
 * @brief 
 * 
 */
class Crash {
  public:
    Crash() = default;
    virtual ~Crash() = default;

    static void crashHandle(int signal) {
      fprintf( stderr, "QGIS died on signal %d\n", signal );
      if ( access( "/usr/bin/gdb", X_OK ) == 0 ) {
        // take full stacktrace using gdb
        // http://stackoverflow.com/questions/3151779/how-its-better-to-invoke-gdb-from-program-to-print-its-stacktrace
        // unfortunately, this is not so simple. the proper method is way more OS-specific
        // than this code would suggest, see http://stackoverflow.com/a/1024937
        char exename[512];
        #if defined(__FreeBSD__)
          int len = readlink( "/proc/curproc/file", exename, sizeof( exename ) - 1 );
        #else
          int len = readlink( "/proc/self/exe", exename, sizeof( exename ) - 1 );
        #endif
        if ( len < 0 ) {
          debugInfo( "Could not read link (%d: %s)\n", errno, strerror( errno ) );
        } else {
          exename[ len ] = 0;

          char pidstr[32];
          snprintf( pidstr, sizeof pidstr, "--pid=%d", getpid() );

          int gdbpid = fork();
          if ( gdbpid == 0 )
          {
            // attach, backtrace and continue
            execl( "/usr/bin/gdb", "gdb", "-q", "-batch", "-n", pidstr, "-ex", "thread", "-ex", "bt full", exename, NULL );
            perror( "cannot exec gdb" );
            exit( 1 );
          } else if ( gdbpid >= 0 ) {
            int status;
            waitpid( gdbpid, &status, 0 );
            debugInfo( "gdb returned %d\n", status );
          } else {
            debugInfo( "Cannot fork (%d: %s)\n", errno, strerror( errno ) );
            dumpBacktrace( 256 );
          }
        }
      } else {
        fprintf( stderr, "cannot find gdb, no backtrace info cout!\n");
      }
      abort();
    }

  private:
    static void debugInfo( const char *fmt, ... ) {
      va_list ap;
      va_start( ap, fmt );
      vfprintf( stderr, fmt, ap );
      va_end( ap );
    }

    static void dumpBacktrace( unsigned int depth ) {
      if ( depth == 0 )
        depth = 20;
      // Below there is a bunch of operations that are not safe in multi-threaded
      // environment (dup()+close() combo, wait(), juggling with file descriptors).
      // Maybe some problems could be resolved with dup2() and waitpid(), but it seems
      // that if the operations on descriptors are not serialized, things will get nasty.
      // That's why there's this lovely mutex here...
      std::lock_guard<std::mutex> guard(crash_lock_);
      int stderr_fd = -1;
      if ( access( "/usr/bin/c++filt", X_OK ) < 0 ) {
        debugInfo( "Stacktrace (c++filt NOT FOUND):\n" );
      } else {
        int fd[2];

        if ( pipe( fd ) == 0 && fork() == 0 ) {
          close( STDIN_FILENO ); // close stdin

          // stdin from pipe
          if ( dup( fd[0] ) != STDIN_FILENO ) {
            std::cout << "dup to stdin failed!" << std::endl;
          }

          close( fd[1] );        // close writing end
          execl( "/usr/bin/c++filt", "c++filt", static_cast< char * >( nullptr ) );
          perror( "could not start c++filt" );
          exit( 1 );
        }

        debugInfo( "Stacktrace (piped through c++filt):\n" );
        stderr_fd = dup( STDERR_FILENO );
        close( fd[0] );          // close reading end
        close( STDERR_FILENO );  // close stderr

        // stderr to pipe
        int stderr_new = dup( fd[1] );
        if ( stderr_new != STDERR_FILENO )
        {
          if ( stderr_new >= 0 )
            close( stderr_new );
          std::cout << "dup to stderr failed!" << std::endl;
        }

        close( fd[1] );  // close duped pipe
      }

      void **buffer = new void *[ depth ];
      int nptrs = backtrace( buffer, depth );
      backtrace_symbols_fd( buffer, nptrs, STDERR_FILENO );
      delete [] buffer;
      if ( stderr_fd >= 0 ) {
        int status;
        close( STDERR_FILENO );
        int dup_stderr = dup( stderr_fd );
        if ( dup_stderr != STDERR_FILENO ) {
          close( dup_stderr );
          std::cout << "dup to stderr failed!" << std::endl;
        }
        close( stderr_fd );
        wait( &status );
      }
    }
};

void initCrashHandle() {
  signal( SIGQUIT, Crash::crashHandle );
  signal( SIGILL, Crash::crashHandle );
  signal( SIGFPE, Crash::crashHandle );
  signal( SIGSEGV,Crash::crashHandle );
  signal( SIGBUS, Crash::crashHandle );
  signal( SIGSYS, Crash::crashHandle );
  signal( SIGTRAP,Crash::crashHandle );
  signal( SIGXCPU,Crash::crashHandle );
  signal( SIGXFSZ, Crash::crashHandle );
}
