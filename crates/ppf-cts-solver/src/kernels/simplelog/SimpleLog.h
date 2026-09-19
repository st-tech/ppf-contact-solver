// File: SimpleLog.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef SIMPLELOG_H
#define SIMPLELOG_H

#include <chrono>
#include <map>
#include <stack>
#include <string>
#include <vector>

class SimpleLog {
  public:
    SimpleLog(std::string name);
    virtual ~SimpleLog();

    void mark(std::string name, double value, bool print = true);
    void record(std::string name, unsigned col, unsigned val,
                bool print = true);
    void push(std::string name);
    void pop(bool print = true);
    void check_empty(std::string file, int line) const;

    static void setPath(std::string data_directory_path);
    static void set(double time);
    static void message(std::string format, ...);

    static double getTime();
    static std::string getLogDirectoryPath();

  private:
    using Time = std::chrono::time_point<std::chrono::steady_clock>;
    std::string m_name;
    Time m_start;
    std::stack<std::pair<Time, std::string>> m_stack;
    std::map<std::string, std::vector<double>> m_dictionary;
};

#endif
