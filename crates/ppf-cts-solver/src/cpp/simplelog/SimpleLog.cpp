// File: SimpleLog.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "SimpleLog.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdarg.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

#define MAX_BUFFER_SIZE 4096
#define SMALL_BUFFER_SIZE 256

static char g_buffer[MAX_BUFFER_SIZE];
static char g_small_buffer[SMALL_BUFFER_SIZE];
static std::string g_data_directory_path;
static int g_depth = 0;
static double g_time = 0.0;

static const char *tstr(uint64_t msec) {
    if (msec < 1000) {
        // Milli seconds
        snprintf(g_small_buffer, SMALL_BUFFER_SIZE, "%lu msec", msec);
    } else if (msec < 1000 * 60 * 3) {
        // Seconds
        snprintf(g_small_buffer, SMALL_BUFFER_SIZE, "%.3f sec", msec / 1000.0);
    } else if (msec < 1000 * 60 * 60 * 3) {
        // Minutes
        snprintf(g_small_buffer, SMALL_BUFFER_SIZE, "%.3f minutes",
                 msec / (60.0 * 1000.0));
    } else if (msec < 1000 * 60 * 60 * 60 * 3) {
        // Hours
        snprintf(g_small_buffer, SMALL_BUFFER_SIZE, "%.3f hours",
                 msec / (60 * 60.0 * 1000.0));
    } else {
        // Days
        snprintf(g_small_buffer, SMALL_BUFFER_SIZE, "%.3f days",
                 msec / (60 * 60.0 * 24 * 1000.0));
    }
    return g_small_buffer;
}

void SimpleLog::setPath(std::string data_directory_path) {
    g_data_directory_path = data_directory_path;
    message("* data_directory_path path = %s", data_directory_path.c_str());
    MKDIR(data_directory_path.c_str());
}

void SimpleLog::set(double time) {
    g_time = time;
    message("* time = %g", time);
}

double SimpleLog::getTime() { return g_time; }

std::string SimpleLog::getLogDirectoryPath() { return g_data_directory_path; }

void SimpleLog::message(std::string format, ...) {
    for (int i = 1; i < g_depth; ++i) {
        printf("   ");
    }
    va_list args;
    va_start(args, format);
    vsnprintf(g_buffer, MAX_BUFFER_SIZE, format.c_str(), args);
    va_end(args);
    printf("%s\n", g_buffer);
    fflush(stdout);
}
//
SimpleLog::SimpleLog(std::string name) {
    std::replace(name.begin(), name.end(), ' ', '_');
    m_name = name;
    g_depth++;
    m_start = std::chrono::steady_clock::now();
    message("====== %s ======", name.c_str());
}

SimpleLog::~SimpleLog() {
    const auto end = std::chrono::steady_clock::now();
    const uint64_t msecs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start)
            .count();
    message("===== %s: %s =====", m_name.c_str(), tstr(msecs));
    const std::string path = g_data_directory_path + "/" + m_name + ".out";
    FILE *fp = fopen(path.c_str(), "a");
    if (fp) {
        fprintf(fp, "%f %lu\n", g_time, (long unsigned int)msecs);
        fclose(fp);
    }
    g_depth--;
    check_empty(m_name, __LINE__);
    if (g_data_directory_path.size()) {
        for (auto it : m_dictionary) {
            const std::string path =
                g_data_directory_path + "/" + m_name + "." + it.first + ".out";
            FILE *fp = fopen(path.c_str(), "a");
            if (fp) {
                for (auto entry : it.second) {
                    if (!fmodf(entry, 1.0)) {
                        fprintf(fp, "%f %d\n", g_time, (int)entry);
                    } else {
                        fprintf(fp, "%f %e\n", g_time, entry);
                    }
                    fflush(stdout);
                }
                fclose(fp);
            }
        }
    }
}

void SimpleLog::check_empty(std::string file, int line) const {
    if (!m_stack.empty()) {
        fprintf(stderr, "ERROR: %s (L%d): stack is not empty\n", file.c_str(),
                line);
        fflush(stderr);
        assert(false);
    }
}

void SimpleLog::mark(std::string name, double value, bool print) {
    std::replace(name.begin(), name.end(), ' ', '_');
    m_dictionary[name].push_back(value);
    if (print) {
        if (!fmodf(value, 1.0)) {
            message("* %s: %d", name.c_str(), (int)value);
        } else {
            message("* %s: %.3e", name.c_str(), value);
        }
    }
}

void SimpleLog::push(std::string name) {
    std::replace(name.begin(), name.end(), ' ', '_');
    m_stack.push(std::make_pair(std::chrono::steady_clock::now(), name));
}

void SimpleLog::pop(bool print) {
    const std::string name = m_stack.top().second;
    const auto end = std::chrono::steady_clock::now();
    const uint64_t msecs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end - m_stack.top().first)
            .count();
    if (print) {
        message("> %s...%s", name.c_str(), tstr(msecs));
    }
    m_stack.pop();
    const std::string path =
        g_data_directory_path + "/" + m_name + "." + name + ".out";
    FILE *fp = fopen(path.c_str(), "a");
    if (fp) {
        fprintf(fp, "%f %lu\n", g_time, (long unsigned int)msecs);
        fclose(fp);
    }
}

void SimpleLog::record(std::string name, unsigned col, unsigned val,
                       bool print) {
    if (print) {
        message("> %s: (%u, %u)", name.c_str(), col, val);
    }
    const std::string path =
        g_data_directory_path + "/" + m_name + "." + name + ".out";
    FILE *fp = fopen(path.c_str(), "a");
    if (fp) {
        fprintf(fp, "%u %u\n", col, val);
        fclose(fp);
    }
}
