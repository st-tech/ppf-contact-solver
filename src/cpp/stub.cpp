// Stub for print_rust function - implemented in Rust, but needed for DLL linking
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" EXPORT void print_rust(const char* msg) {}
