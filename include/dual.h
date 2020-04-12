// Description: Class for scalar, dual numbers with operator overloads
//              to enable forward-mode automatic differentiation
// Author: Jerel Nielsen
// Date: 20 Sep 2019
#pragma once

#include <cmath>
#include <iostream>

namespace dual
{

template<typename T = double>
class Dual
{
public:
    Dual() : value(T(0)), derivative(T(0)) {}
    Dual(const T& _value) : value(_value), derivative(T(0)) {}
    Dual(const T& _value, const T& _derivative) : value(_value), derivative(_derivative) {}
    Dual(const Dual& other)
    {
        value = other.value;
        derivative = other.derivative;
    }
    ~Dual() {}

    void inline setValue(const T& _value) { value = _value; }
    void inline setDerivative(const T& _derivative) { derivative = _derivative; }
    const T inline v() const { return value; }
    const T inline dv() const { return derivative; }

    // Operator overloads
    Dual operator+(const Dual& other)
    {
        return Dual(value + other.value, derivative + other.derivative);
    }
    Dual operator-(const Dual& other)
    {
        return Dual(value - other.value, derivative - other.derivative);
    }
    Dual operator*(const Dual& other)
    {
        return Dual(value * other.value, value * other.derivative + derivative * other.value);
    }
    Dual operator/(const Dual& other)
    {
        return Dual(value / other.value, (derivative * other.value - value * other.derivative) / (other.value * other.value));
    }
    friend std::ostream& operator<<(std::ostream& os, const Dual& num)
    {
        os << "(" << num.value << ", " << num.derivative << ")";
        return os;
    }

    // Additional basic functions
    friend Dual sin(const Dual& x1)
    {
        return Dual(sin(x1.value), x1.derivative * cos(x1.value));
    }
    friend Dual cos(const Dual& x1)
    {
        return Dual(cos(x1.value), -x1.derivative * sin(x1.value));
    }
    friend Dual exp(const Dual& x1)
    {
        return Dual(exp(x1.value), x1.derivative * exp(x1.value));
    }
    friend Dual log(const Dual& x1)
    {
        return Dual(log(x1.value), x1.derivative / x1.value);
    }
    friend Dual pow(const Dual& x1, const T& k)
    {
        return Dual(pow(x1.value, k), k * pow(x1.value, k-1) * x1.derivative);
    }
    friend Dual abs(const Dual& x1)
    {
        return Dual(abs(x1.value), x1.derivative * x1.value / abs(x1.value));
    }

private:
    T value; // Regular number value
    T derivative; // Derivative at value
};

} // namespace dual