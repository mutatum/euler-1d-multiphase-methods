#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

class TestFramework
{
private:
    std::vector<std::function<void()>> tests_;
    std::vector<std::string> test_names_;
    static inline int total_tests_ = 0;
    static inline int passed_tests_ = 0;
    static inline int failed_tests_ = 0;

public:
    void add_test(const std::string &name, std::function<void()> test)
    {
        test_names_.push_back(name);
        tests_.push_back(std::move(test));
    }

    void run_all_tests()
    {
        std::cout << "Running " << tests_.size() << " tests...\n";
        std::cout << std::string(60, '=') << "\n";

        for (size_t i = 0; i < tests_.size(); ++i)
        {
            std::cout << "[" << std::setw(3) << i + 1 << "/" << tests_.size() << "] "
                      << test_names_[i] << "... ";

            try
            {
                tests_[i]();
                std::cout << "PASSED\n";
                ++passed_tests_;
            }
            catch (const std::exception &e)
            {
                std::cout << "FAILED: " << e.what() << "\n";
                ++failed_tests_;
            }
            ++total_tests_;
        }

        std::cout << std::string(60, '=') << "\n";
        std::cout << "Results: " << passed_tests_ << " passed, "
                  << failed_tests_ << " failed, "
                  << total_tests_ << " total\n";
    }

    static void assert_true(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("Assertion failed: " + message);
        }
    }

    static void assert_false(bool condition, const std::string &message)
    {
        if (condition)
        {
            throw std::runtime_error("Assertion failed: " + message);
        }
    }

    template <typename T>
    static void assert_equal(const T &expected, const T &actual, const std::string &message)
    {
        if (expected != actual)
        {
            std::ostringstream oss;
            oss << "Assertion failed: " << message
                << " (expected: " << expected << ", actual: " << actual << ")";
            throw std::runtime_error(oss.str());
        }
    }

    template <typename T, typename U>
    static void assert_equal(const T &expected, const U &actual, const std::string &message)
    {
        if (static_cast<T>(actual) != expected)
        {
            throw std::runtime_error("Assertion failed: " + message +
                                     " (expected != actual)");
        }
    }

    template <typename T>
    static void assert_near(const T &expected, const T &actual, const T &tolerance,
                            const std::string &message)
    {
        if (std::abs(expected - actual) > tolerance)
        {
            std::cout << std::endl
                      << "Expected: " << expected << ", Actual: " << actual << ", Tolerance: " << tolerance << std::endl;
            throw std::runtime_error("Assertion failed: " + message +
                                     " (|expected - actual| > tolerance)");
        }
    }

    static void assert_throws(std::function<void()> func, const std::string &message)
    {
        try
        {
            func();
            throw std::runtime_error("Expected exception but none was thrown: " + message);
        }
        catch (const std::exception &e)
        {
            // std::cout << message;
            if (std::string(e.what()).find("Expected exception") != std::string::npos)
            {
                throw; // Re-throw if it's our assertion failure
            }
            // Exception was thrown as expected
        }
    }
};

// Remove static definitions since we're using inline static
// int TestFramework::total_tests_ = 0;
// int TestFramework::passed_tests_ = 0;
// int TestFramework::failed_tests_ = 0;

#define TEST(test_name)                                                        \
    void test_##test_name();                                                   \
    struct test_##test_name##_registrar                                        \
    {                                                                          \
        test_##test_name##_registrar()                                         \
        {                                                                      \
            extern TestFramework global_test_framework;                        \
            global_test_framework.add_test(#test_name, test_##test_name);      \
        }                                                                      \
    };                                                                         \
    static test_##test_name##_registrar test_##test_name##_registrar_instance; \
    void test_##test_name()

#define ASSERT_TRUE(condition) TestFramework::assert_true(condition, #condition)
#define ASSERT_FALSE(condition) TestFramework::assert_false(condition, #condition)
#define ASSERT_EQ(expected, actual) \
    do { \
        auto exp_val = (expected); \
        auto act_val = (actual); \
        if (exp_val != act_val) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #expected " == " #actual \
                << " (expected: " << exp_val << ", actual: " << act_val << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)
#define ASSERT_NEAR(expected, actual, tolerance) TestFramework::assert_near(expected, actual, tolerance, #expected " ≈ " #actual)
#define ASSERT_THROWS(func) TestFramework::assert_throws([&]() { func; }, #func)
