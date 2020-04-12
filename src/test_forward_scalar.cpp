// Description: Tests for the forward-mode scalar version.
// Author: Jerel Nielsen
// Date: 20 Sep 2019

#include <iostream>
#include <gtest/gtest.h>
#include "dual.h"
#include "dual2.h"

#define NUM_ITERS 1e8

namespace dual
{

TEST(DualNumbers, Constructors)
{
    Dual<> x1;
    EXPECT_FLOAT_EQ(x1.v(), 0.0);
    EXPECT_FLOAT_EQ(x1.dv(), 0.0);

    Dual<> x2(4.56);
    EXPECT_FLOAT_EQ(x2.v(), 4.56);
    EXPECT_FLOAT_EQ(x2.dv(), 0.0);

    Dual<> x3(7.53, 2.99);
    EXPECT_FLOAT_EQ(x3.v(), 7.53);
    EXPECT_FLOAT_EQ(x3.dv(), 2.99);

    Dual<> x4(x3);
    EXPECT_FLOAT_EQ(x4.v(), x3.v());
    EXPECT_FLOAT_EQ(x4.dv(), x3.dv()); }

TEST(DualNumbers, Addition)
{
    Dual<> x1(1.2, 2.9);
    Dual<> x2(9.1, 7.5);
    auto x3 = x1 + x2;
    EXPECT_FLOAT_EQ(x3.v(), 10.3);
    EXPECT_FLOAT_EQ(x3.dv(), 10.4);
}

TEST(DualNumbers, Subtraction)
{
    Dual<> x1(1.2, 2.9);
    Dual<> x2(9.1, 7.5);
    auto x3 = x1 - x2;
    EXPECT_FLOAT_EQ(x3.v(), -7.9);
    EXPECT_FLOAT_EQ(x3.dv(), -4.6);
}

TEST(DualNumbers, Multiplication)
{
    Dual<> x1(6, 10);
    Dual<> x2(3, 5);
    auto x3 = x1 * x2;
    EXPECT_FLOAT_EQ(x3.v(), 18);
    EXPECT_FLOAT_EQ(x3.dv(), 60);
}

TEST(DualNumbers, Division)
{
    Dual<> x1(6, 10);
    Dual<> x2(3, 2);
    auto x3 = x1 / x2;
    EXPECT_FLOAT_EQ(x3.v(), 2);
    EXPECT_FLOAT_EQ(x3.dv(), 2);
}

TEST(DualNumbers, Sine)
{
    Dual<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = sin(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * cos(x1.v() * x1.v()));
}

TEST(DualNumbers, Cosine)
{
    Dual<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = cos(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), -2 * x1.v() * sin(x1.v() * x1.v()));
}

TEST(DualNumbers, Exponential)
{
    Dual<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = exp(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * exp(x1.v() * x1.v()));
}

TEST(DualNumbers, Logarithm)
{
    Dual<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = log(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() / (x1.v() * x1.v()));
}

TEST(DualNumbers, Power)
{
    Dual<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = x1 * x1 * x1;
    EXPECT_FLOAT_EQ(f.dv(), 3 * x1.v() * x1.v());
}

TEST(DualNumbers, Absolute)
{
    Dual<> x1(-5.32, 1); // Differentiate w.r.t. this one
    Dual<> f = abs(x1 * x1 - 2.3);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * (x1.v() * x1.v() - 2.3) / abs(x1.v() * x1.v() - 2.3));
}

TEST(DualNumbers, SpeedWithoutExpressions)
{
    for (int i = 0; i < NUM_ITERS; ++i)
    {
        Dual<> x1(2.3, 1);
        Dual<> f = exp(sin(cos(log(x1 * x1))));
        EXPECT_FLOAT_EQ(f.dv(), exp(sin(cos(log(x1.v() * x1.v())))) * cos(cos(log(x1.v() * x1.v()))) * -sin(log(x1.v() * x1.v())) * 2 * x1.v() / (x1.v() * x1.v()));
    }
}

} // namespace dual

namespace dual2
{

TEST(DualNumbers2, Constructors)
{
    Dual2<> x1;
    EXPECT_FLOAT_EQ(x1.v(), 0.0);
    EXPECT_FLOAT_EQ(x1.dv(), 0.0);

    Dual2<> x2(4.56);
    EXPECT_FLOAT_EQ(x2.v(), 4.56);
    EXPECT_FLOAT_EQ(x2.dv(), 0.0);

    Dual2<> x3(7.53, 2.99);
    EXPECT_FLOAT_EQ(x3.v(), 7.53);
    EXPECT_FLOAT_EQ(x3.dv(), 2.99);

    Dual2<> x4(x3);
    EXPECT_FLOAT_EQ(x4.v(), x3.v());
    EXPECT_FLOAT_EQ(x4.dv(), x3.dv()); }

TEST(DualNumbers2, Addition)
{
    Dual2<> x1(1.2, 2.9);
    Dual2<> x2(9.1, 7.5);
    auto x3 = (x1 + x2).eval();
    EXPECT_FLOAT_EQ(x3.v(), 10.3);
    EXPECT_FLOAT_EQ(x3.dv(), 10.4);
}

TEST(DualNumbers2, Subtraction)
{
    Dual2<> x1(1.2, 2.9);
    Dual2<> x2(9.1, 7.5);
    auto x3 = (x1 - x2).eval();
    EXPECT_FLOAT_EQ(x3.v(), -7.9);
    EXPECT_FLOAT_EQ(x3.dv(), -4.6);
}

TEST(DualNumbers2, Multiplication)
{
    Dual2<> x1(6, 10);
    Dual2<> x2(3, 5);
    auto x3 = (x1 * x2).eval();
    EXPECT_FLOAT_EQ(x3.v(), 18);
    EXPECT_FLOAT_EQ(x3.dv(), 60);
}

TEST(DualNumbers2, Division)
{
    Dual2<> x1(6, 10);
    Dual2<> x2(3, 2);
    auto x3 = (x1 / x2).eval();
    EXPECT_FLOAT_EQ(x3.v(), 2);
    EXPECT_FLOAT_EQ(x3.dv(), 2);
}

TEST(DualNumbers2, Sine)
{
    Dual2<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = sin(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * std::cos(x1.v() * x1.v()));
}

TEST(DualNumbers2, Cosine)
{
    Dual2<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = cos(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), -2 * x1.v() * std::sin(x1.v() * x1.v()));
}

TEST(DualNumbers2, Exponential)
{
    Dual2<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = exp(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * std::exp(x1.v() * x1.v()));
}

TEST(DualNumbers2, Logarithm)
{
    Dual2<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = log(x1 * x1);
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() / (x1.v() * x1.v()));
}

TEST(DualNumbers2, Power)
{
    Dual2<> x1(5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = x1 * x1 * x1;
    EXPECT_FLOAT_EQ(f.dv(), 3 * x1.v() * x1.v());
}

TEST(DualNumbers2, Absolute)
{
    Dual2<> x1(-5.32, 1); // Differentiate w.r.t. this one
    Dual2<> f;
    f = abs(x1 * x1 - Dual2<>(2.3,0));
    EXPECT_FLOAT_EQ(f.dv(), 2 * x1.v() * (x1.v() * x1.v() - 2.3) / std::abs(x1.v() * x1.v() - 2.3));
}

TEST(DualNumbers2, SpeedWithExpressions)
{
    for (int i = 0; i < NUM_ITERS; ++i)
    {
        Dual2<> x1(2.3, 1);
        Dual2<> f;
        f = exp(sin(cos(log(x1 * x1))));
        EXPECT_FLOAT_EQ(f.dv(), std::exp(std::sin(std::cos(std::log(x1.v() * x1.v())))) * std::cos(std::cos(std::log(x1.v() * x1.v()))) * -std::sin(std::log(x1.v() * x1.v())) * 2 * x1.v() / (x1.v() * x1.v()));
    }
}

} // namespace dual2